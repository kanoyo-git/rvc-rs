use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc,
};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use rtrb::{Consumer, Producer, RingBuffer};
use serde::Serialize;

/// Audio and model worker boundaries share a frame count but have independent
/// callback periods. Neither audio callback runs model inference or takes a lock.
const BUFFERED_BLOCKS: usize = 2;
const POLL_INTERVAL: Duration = Duration::from_millis(1);

#[derive(Clone, Copy)]
struct InputFrame {
    position: u64,
    sample: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct WorkerStats {
    pub processed_blocks: u64,
    pub dropped_input_frames: u64,
    pub dropped_output_blocks: u64,
    pub silent_output_frames: u64,
    pub last_inference_us: u64,
}

#[derive(Default)]
struct Counters {
    processed_blocks: AtomicU64,
    dropped_input_frames: AtomicU64,
    dropped_output_blocks: AtomicU64,
    silent_output_frames: AtomicU64,
    inference_us: AtomicU64,
}

#[derive(Default)]
struct SharedState {
    stopping: AtomicBool,
    counters: Counters,
    error: OnceLock<anyhow::Error>,
}

pub struct Capture {
    producer: Producer<InputFrame>,
    position: u64,
    state: Arc<SharedState>,
}

impl Capture {
    /// Push mono input frames. A full queue drops new samples and accounts for
    /// the gap, so a later model call can reset its rolling context.
    pub fn write<I>(&mut self, samples: I)
    where
        I: IntoIterator<Item = f32>,
        I::IntoIter: ExactSizeIterator,
    {
        if self.state.stopping.load(Ordering::Relaxed) {
            return;
        }
        let samples = samples.into_iter();
        let count = samples.len();
        let start = self.position;
        self.position = start.wrapping_add(count as u64);
        let available = self.producer.slots().min(count);
        let written = self
            .producer
            .write_chunk_uninit(available)
            .map_or(0, |chunk| {
                chunk.fill_from_iter(samples.take(available).enumerate().map(|(i, sample)| {
                    InputFrame {
                        position: start.wrapping_add(i as u64),
                        sample: if sample.is_finite() { sample } else { 0.0 },
                    }
                }))
            });
        let dropped = count - written;
        if dropped > 0 {
            self.state
                .counters
                .dropped_input_frames
                .fetch_add(dropped as u64, Ordering::Relaxed);
        }
    }
}

pub struct Playback {
    consumer: Consumer<f32>,
    state: Arc<SharedState>,
}

impl Playback {
    /// Call the sink once per frame. It can copy each mono sample into as many
    /// channels as the host needs. An empty queue is filled with silence.
    pub fn read(&mut self, frames: usize, mut write: impl FnMut(f32)) {
        if self.state.stopping.load(Ordering::Relaxed) {
            for _ in 0..frames {
                write(0.0);
            }
            return;
        }
        let requested = self.consumer.slots().min(frames);
        let available = self.consumer.read_chunk(requested).map_or(0, |chunk| {
            let (left, right) = chunk.as_slices();
            for &sample in left.iter().chain(right) {
                write(sample);
            }
            chunk.commit_all();
            requested
        });
        for _ in available..frames {
            write(0.0);
        }
        if available < frames {
            self.state
                .counters
                .silent_output_frames
                .fetch_add((frames - available) as u64, Ordering::Relaxed);
        }
    }
}

pub struct AudioWorker {
    state: Arc<SharedState>,
    thread: Option<JoinHandle<()>>,
}

impl AudioWorker {
    /// `process` receives an exact contiguous model block and a reusable
    /// output buffer of the same length. It should reset rolling state when
    /// the third argument marks a gap caused by input overflow.
    pub fn start<F>(block_frames: usize, process: F) -> Result<(Self, Capture, Playback)>
    where
        F: FnMut(&[f32], &mut [f32], bool) -> Result<()> + Send + 'static,
    {
        Self::spawn(block_frames, process, false)
    }

    /// Warm the model on its own worker before opening audio streams.
    pub fn start_ready<F>(block_frames: usize, process: F) -> Result<(Self, Capture, Playback)>
    where
        F: FnMut(&[f32], &mut [f32], bool) -> Result<()> + Send + 'static,
    {
        Self::spawn(block_frames, process, true)
    }

    fn spawn<F>(block_frames: usize, process: F, warmup: bool) -> Result<(Self, Capture, Playback)>
    where
        F: FnMut(&[f32], &mut [f32], bool) -> Result<()> + Send + 'static,
    {
        ensure!(
            (1..=768_000).contains(&block_frames),
            "invalid audio block size {block_frames}"
        );
        let (capture_producer, capture_consumer) = RingBuffer::new(block_frames * BUFFERED_BLOCKS);
        let (playback_producer, playback_consumer) =
            RingBuffer::new(block_frames * BUFFERED_BLOCKS);
        let state = Arc::new(SharedState::default());
        let worker_state = Arc::clone(&state);
        let (ready, prepared) = warmup.then(|| mpsc::sync_channel(1)).unzip();
        let thread = Builder::new()
            .name("rvc-audio-worker".into())
            .spawn(move || {
                run(
                    block_frames,
                    process,
                    worker_state,
                    capture_consumer,
                    playback_producer,
                    ready,
                );
            })
            .context("start audio worker")?;
        let worker = Self {
            state: Arc::clone(&state),
            thread: Some(thread),
        };
        if let Some(prepared) = prepared {
            prepared
                .recv()
                .context("audio worker stopped during startup")?
                .context("preparing the realtime model")?;
        }
        let capture = Capture {
            producer: capture_producer,
            position: 0,
            state: Arc::clone(&state),
        };
        let playback = Playback {
            consumer: playback_consumer,
            state,
        };
        Ok((worker, capture, playback))
    }

    pub fn check(&self) -> Result<()> {
        if let Some(error) = self.state.error.get() {
            return Err(anyhow!("audio processing failed: {error:#}"));
        }
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(anyhow!("audio worker stopped unexpectedly"));
        }
        Ok(())
    }

    pub fn stats(&self) -> WorkerStats {
        let counters = &self.state.counters;
        WorkerStats {
            processed_blocks: counters.processed_blocks.load(Ordering::Relaxed),
            dropped_input_frames: counters.dropped_input_frames.load(Ordering::Relaxed),
            dropped_output_blocks: counters.dropped_output_blocks.load(Ordering::Relaxed),
            silent_output_frames: counters.silent_output_frames.load(Ordering::Relaxed),
            last_inference_us: counters.inference_us.load(Ordering::Relaxed),
        }
    }

    pub fn stop(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        self.state.stopping.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            thread
                .join()
                .map_err(|_| anyhow!("audio worker panicked"))?;
        }
        self.check()
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        // Explicit stop reports model errors. Dropping during startup or
        // unwinding must still release the worker.
        let _ = self.join();
    }
}

fn run<F>(
    block_frames: usize,
    mut process: F,
    state: Arc<SharedState>,
    mut capture: Consumer<InputFrame>,
    mut playback: Producer<f32>,
    ready: Option<mpsc::SyncSender<Result<()>>>,
) where
    F: FnMut(&[f32], &mut [f32], bool) -> Result<()>,
{
    let mut input = vec![0.0; block_frames];
    let mut output = vec![0.0; block_frames];
    let mut discontinuity = ready.is_some();
    if let Some(ready) = ready {
        let result = process(&input, &mut output, true);
        let proceed = result.is_ok();
        if ready.send(result).is_err() || !proceed {
            return;
        }
    }
    let mut filled = 0;
    let mut expected = 0u64;
    while !state.stopping.load(Ordering::Acquire) {
        // The playback callback can only free space. Wait for room for the
        // complete result so a brief render delay does not discard a block.
        if playback.slots() < block_frames {
            std::thread::park_timeout(POLL_INTERVAL);
            continue;
        }
        let available = capture.slots();
        for _ in 0..available {
            let Ok(frame) = capture.pop() else {
                break;
            };
            if frame.position != expected {
                filled = 0;
                discontinuity = true;
            }
            expected = frame.position.wrapping_add(1);
            input[filled] = frame.sample;
            filled += 1;
            if filled == block_frames {
                break;
            }
        }
        if filled < block_frames {
            std::thread::park_timeout(POLL_INTERVAL);
            continue;
        }

        let start = Instant::now();
        output.fill(0.0);
        if let Err(error) = process(&input, &mut output, discontinuity) {
            let _ = state.error.set(error);
            state.stopping.store(true, Ordering::Release);
            break;
        }
        state
            .counters
            .inference_us
            .store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        state
            .counters
            .processed_blocks
            .fetch_add(1, Ordering::Relaxed);

        for sample in &mut output {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        // Publish the complete block at once. An overlapping render callback
        // cannot observe and play a partial model result.
        if playback.push_entire_slice(&output).is_err() {
            state
                .counters
                .dropped_output_blocks
                .fetch_add(1, Ordering::Relaxed);
        }
        filled = 0;
        discontinuity = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_callback_periods_produce_exact_model_blocks() -> Result<()> {
        let (worker, mut capture, mut playback) =
            AudioWorker::start(4, |input, output, discontinuity| {
                assert!(!discontinuity);
                for (result, sample) in output.iter_mut().zip(input) {
                    *result = 2.0 * sample;
                }
                Ok(())
            })?;
        capture.write([1.0, 2.0, 3.0]);
        capture.write([4.0, 5.0]);
        let deadline = Instant::now() + Duration::from_secs(1);
        while playback.consumer.slots() < 4 {
            worker.check()?;
            ensure!(Instant::now() < deadline, "timed out waiting for audio");
            std::thread::sleep(POLL_INTERVAL);
        }
        let mut decoded = Vec::new();
        playback.read(3, |sample| decoded.push(sample));
        playback.read(3, |sample| decoded.push(sample));
        assert_eq!(decoded, [2.0, 4.0, 6.0, 8.0, 0.0, 0.0]);
        assert_eq!(worker.stats().silent_output_frames, 2);
        worker.stop()
    }

    #[test]
    fn capture_gap_drops_partial_history_and_signals_model() -> Result<()> {
        let (event, events) = std::sync::mpsc::channel();
        let (worker, mut capture, _playback) =
            AudioWorker::start(3, move |input, _, discontinuity| {
                event.send((input.to_vec(), discontinuity))?;
                Ok(())
            })?;
        capture.write([1.0, 2.0, 3.0]);
        assert_eq!(
            events.recv_timeout(Duration::from_secs(1))?,
            (vec![1.0, 2.0, 3.0], false)
        );
        capture.write([4.0]);
        // Advance the capture frame counter to emulate input overflow.
        capture.position += 2;
        capture.write([7.0, 8.0, 9.0]);
        assert_eq!(
            events.recv_timeout(Duration::from_secs(1))?,
            (vec![7.0, 8.0, 9.0], true)
        );
        worker.stop()
    }

    #[test]
    fn brief_playback_lag_preserves_the_next_block() -> Result<()> {
        let (rendered, events) = mpsc::channel();
        let (worker, mut capture, mut playback) =
            AudioWorker::start(2, move |input, output, _| {
                output.copy_from_slice(input);
                rendered.send(())?;
                Ok(())
            })?;
        let wait_until_full = |playback: &Playback| -> Result<()> {
            let deadline = Instant::now() + Duration::from_secs(1);
            while playback.consumer.slots() < 4 {
                worker.check()?;
                ensure!(Instant::now() < deadline, "timed out waiting for playback");
                std::thread::sleep(POLL_INTERVAL);
            }
            Ok(())
        };

        capture.write([1.0, 2.0, 3.0, 4.0]);
        wait_until_full(&playback)?;
        assert_eq!(events.try_iter().count(), 2);

        capture.write([5.0, 6.0]);
        let mut received = Vec::new();
        playback.read(1, |sample| received.push(sample));
        assert!(matches!(
            events.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        playback.read(1, |sample| received.push(sample));
        wait_until_full(&playback)?;
        playback.read(4, |sample| received.push(sample));
        assert_eq!(received, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(worker.stats().dropped_output_blocks, 0);
        worker.stop()
    }
}
