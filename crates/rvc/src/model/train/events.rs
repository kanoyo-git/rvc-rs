use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use prost::Message;
use tch::{Device, Kind, Tensor};

pub(super) struct Events(BufWriter<File>);

impl Events {
    pub(super) fn new(directory: &Path) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let path = directory.join(format!(
            "events.out.tfevents.{}.rvc.{}.{}",
            now.as_secs(),
            std::process::id(),
            now.subsec_nanos()
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating training event file {}", path.display()))?;
        let mut writer = Self(BufWriter::new(file));
        let mut version = event_time()?;
        bytes(&mut version, 3, b"brain.Event:2");
        writer.record(&version)?;
        Ok(writer)
    }

    pub(super) fn summary(
        &mut self,
        step: usize,
        scalars: &[(String, f64)],
        images: &[(&str, Tensor)],
    ) -> Result<()> {
        let mut summary = Vec::new();
        for (tag, scalar) in scalars {
            let mut value = Vec::new();
            bytes(&mut value, 1, tag.as_bytes());
            value.push(2 << 3 | 5); // Summary.Value.simple_value, fixed32.
            value.extend_from_slice(&(*scalar as f32).to_le_bytes());
            bytes(&mut summary, 1, &value);
        }
        for (tag, tensor) in images {
            let (height, width) = tensor.size2()?;
            ensure!(height > 0 && width > 0, "empty training spectrogram image");
            let cells = Vec::<f32>::try_from(
                tensor
                    .f_detach()?
                    .f_to_device(Device::Cpu)?
                    .f_to_kind(Kind::Float)?
                    .f_reshape([-1])?,
            )?;
            let png = spectrogram_png(&cells, width.try_into()?, height.try_into()?);
            let mut image = Vec::new();
            integer(&mut image, 1, height as u64);
            integer(&mut image, 2, width as u64);
            integer(&mut image, 3, 1);
            bytes(&mut image, 4, &png);
            let mut value = Vec::new();
            bytes(&mut value, 1, tag.as_bytes());
            bytes(&mut value, 4, &image);
            bytes(&mut summary, 1, &value);
        }
        let mut event = event_time()?;
        integer(&mut event, 2, step as u64);
        bytes(&mut event, 5, &summary);
        self.record(&event)
    }

    fn record(&mut self, event: &[u8]) -> Result<()> {
        let len = (event.len() as u64).to_le_bytes();
        self.0.write_all(&len)?;
        self.0.write_all(&masked_crc32c(&len).to_le_bytes())?;
        self.0.write_all(event)?;
        self.0.write_all(&masked_crc32c(event).to_le_bytes())?;
        self.0.flush()?;
        Ok(())
    }
}

#[derive(Message)]
struct ScalarEvent {
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(message, optional, tag = "5")]
    summary: Option<ScalarSummary>,
}

#[derive(Message)]
struct ScalarSummary {
    #[prost(message, repeated, tag = "1")]
    values: Vec<ScalarValue>,
}

#[derive(Message)]
struct ScalarValue {
    #[prost(string, tag = "1")]
    tag: String,
    #[prost(float, optional, tag = "2")]
    number: Option<f32>,
}

pub(crate) fn paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("events.out.tfevents.")
            && entry.path().is_file()
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

type Point = (i64, f32);

#[derive(Default)]
struct ScalarCurve {
    points: Vec<Point>,
    // First record and step of each surviving run when a checkpoint is resumed.
    runs: VecDeque<(u64, i64)>,
    sampled: bool,
    replay: bool,
}

impl ScalarCurve {
    fn observe(&mut self, ordinal: u64, point: Point) {
        let rewind = self
            .points
            .last()
            .is_some_and(|previous| previous.0 >= point.0);
        if rewind || self.runs.is_empty() {
            while self.runs.back().is_some_and(|run| run.1 >= point.0) {
                self.runs.pop_back();
            }
            self.runs.push_back((ordinal, point.0));
        }
        if rewind {
            self.replay |= self.sampled;
            self.points
                .truncate(self.points.partition_point(|p| p.0 < point.0));
        }
        self.push(point);
    }

    fn survives_rewind(&mut self, ordinal: u64, step: i64) -> bool {
        while self.runs.get(1).is_some_and(|next| next.0 <= ordinal) {
            self.runs.pop_front();
        }
        self.runs.front().is_some_and(|run| ordinal >= run.0)
            && self.runs.get(1).is_none_or(|next| step < next.1)
    }

    fn push(&mut self, point: Point) {
        self.points.push(point);
        if self.points.len() <= 600 {
            return;
        }
        self.sampled = true;
        let first = self.points[0];
        // Coarser power-of-two step buckets align with all earlier buckets.
        // Their minima and maxima still represent all original observations.
        let mut shift = 0;
        while i128::from(point.0 >> shift) - i128::from(first.0 >> shift) > 149 {
            shift += 1;
        }
        let mut sampled = vec![first];
        for bucket in self
            .points
            .chunk_by(|a, b| (a.0 >> shift) == (b.0 >> shift))
        {
            let (mut low, mut high) = (bucket[0], bucket[0]);
            for &value in &bucket[1..] {
                if value.1 < low.1 {
                    low = value;
                }
                if value.1 > high.1 {
                    high = value;
                }
            }
            if low.0 > high.0 {
                std::mem::swap(&mut low, &mut high);
            }
            for value in [low, high] {
                if sampled.last() != Some(&value) {
                    sampled.push(value);
                }
            }
        }
        if sampled.last() != Some(&point) {
            sampled.push(point);
        }
        self.points = sampled;
    }
}

pub(crate) fn scalars(directory: &Path) -> Result<BTreeMap<String, VecDeque<Point>>> {
    let mut files = Vec::new();
    for path in paths(directory)? {
        let size = fs::metadata(&path)?.len();
        files.push((path, size));
    }

    let mut series = BTreeMap::<String, ScalarCurve>::new();
    visit_scalars(&files, |ordinal, tag, point| {
        if series.len() < 64 || series.contains_key(&tag) {
            series.entry(tag).or_default().observe(ordinal, point);
        }
    })?;
    if series.values().any(|curve| curve.replay) {
        // A rewind can cut through a sampled bucket. Replay the surviving runs
        // to recover extrema that were hidden by now-overwritten observations.
        for curve in series.values_mut().filter(|curve| curve.replay) {
            curve.points.clear();
        }
        visit_scalars(&files, |ordinal, tag, point| {
            if let Some(curve) = series.get_mut(&tag)
                && curve.replay
                && curve.survives_rewind(ordinal, point.0)
            {
                curve.push(point);
            }
        })?;
    }
    Ok(series
        .into_iter()
        .map(|(tag, curve)| (tag, curve.points.into()))
        .collect())
}

fn visit_scalars(
    files: &[(PathBuf, u64)],
    mut visit: impl FnMut(u64, String, Point),
) -> Result<()> {
    let mut ordinal = 0;
    for (path, size) in files {
        let mut reader = BufReader::new(File::open(path)?.take(*size));
        let mut bytes = Vec::new();
        while let Some(record) = read_record(&mut reader, &mut bytes)
            .with_context(|| format!("reading {}", path.display()))?
        {
            let event = ScalarEvent::decode(record)?;
            if let Some(summary) = event.summary {
                for value in summary.values {
                    if let Some(number) = value.number.filter(|number| number.is_finite()) {
                        visit(ordinal, value.tag, (event.step, number));
                        ordinal += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

fn read_record<'a>(reader: &mut impl Read, bytes: &'a mut Vec<u8>) -> Result<Option<&'a [u8]>> {
    let mut header = [0u8; 12];
    let mut crc = [0u8; 4];
    let mut read = |buffer: &mut [u8]| match reader.read_exact(buffer) {
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Ok(false),
        result => result.map(|()| true),
    };
    if !read(&mut header)? {
        return Ok(None);
    }
    let (size, header_crc) = header.split_at(8);
    ensure!(
        masked_crc32c(size) == u32::from_le_bytes(header_crc.try_into()?),
        "invalid event header checksum"
    );
    let length = u64::from_le_bytes(size.try_into()?);
    ensure!(length <= 32 * 1024 * 1024, "training event exceeds 32 MiB");
    bytes.resize(length as usize, 0);
    if !read(bytes)? || !read(&mut crc)? {
        return Ok(None);
    }
    ensure!(
        masked_crc32c(bytes) == u32::from_le_bytes(crc),
        "invalid event checksum"
    );
    Ok(Some(bytes))
}

fn event_time() -> Result<Vec<u8>> {
    let mut event = vec![1 << 3 | 1]; // Event.wall_time, fixed64.
    event.extend_from_slice(
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs_f64()
            .to_le_bytes(),
    );
    Ok(event)
}

fn bytes(output: &mut Vec<u8>, field: u8, payload: &[u8]) {
    output.push(field << 3 | 2);
    varint(output, payload.len() as u64);
    output.extend_from_slice(payload);
}

fn integer(output: &mut Vec<u8>, field: u8, number: u64) {
    output.push(field << 3);
    varint(output, number);
}

fn varint(output: &mut Vec<u8>, mut number: u64) {
    while number >= 128 {
        output.push(number as u8 | 0x80);
        number >>= 7;
    }
    output.push(number as u8);
}

fn masked_crc32c(payload: &[u8]) -> u32 {
    crc32(payload, 0x82f6_3b78)
        .rotate_right(15)
        .wrapping_add(0xa282_ead8)
}

fn crc32(payload: &[u8], polynomial: u32) -> u32 {
    let mut crc = !0u32;
    for &byte in payload {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let bit = crc & 1;
            crc = (crc >> 1) ^ (0u32.wrapping_sub(bit) & polynomial);
        }
    }
    !crc
}

fn spectrogram_png(cells: &[f32], width: u32, height: u32) -> Vec<u8> {
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for &value in cells.iter().filter(|value| value.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    let mut pixels = Vec::with_capacity(cells.len() + height as usize);
    for row in cells.chunks_exact(width as usize).rev() {
        pixels.push(0); // PNG unfiltered scanline.
        pixels.extend(row.iter().map(|&value| {
            if high > low && value.is_finite() {
                (255.0 * (value - low) / (high - low)).round() as u8
            } else {
                0
            }
        }));
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 0, 0, 0, 0]); // Eight-bit grayscale.
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &zlib(&pixels));
    png_chunk(&mut png, b"IEND", &[]);
    png
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    png.extend_from_slice(&crc32(&png[start..], 0xedb8_8320).to_be_bytes());
}

fn zlib(data: &[u8]) -> Vec<u8> {
    let mut encoded = vec![0x78, 1];
    let mut blocks = data.chunks(u16::MAX as usize).peekable();
    while let Some(block) = blocks.next() {
        encoded.push(u8::from(blocks.peek().is_none())); // Stored deflate block.
        let len = block.len() as u16;
        encoded.extend_from_slice(&len.to_le_bytes());
        encoded.extend_from_slice(&(!len).to_le_bytes());
        encoded.extend_from_slice(block);
    }
    let (mut first, mut second) = (1u32, 0u32);
    for &byte in data {
        first = (first + u32::from(byte)) % 65_521;
        second = (second + first) % 65_521;
    }
    encoded.extend_from_slice(&((second << 16) | first).to_be_bytes());
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksums_match_the_standard_test_vector() {
        assert_eq!(crc32(b"123456789", 0x82f6_3b78), 0xe306_9283);
        assert_eq!(crc32(b"123456789", 0xedb8_8320), 0xcbf4_3926);
    }

    #[test]
    fn long_resumed_curves_keep_endpoints_and_surviving_extrema() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut events = Events::new(directory.path())?;
        for epoch in 1..=10_100 {
            let loss = match epoch {
                189 => -9.0,
                194 => -100.0,
                201 => 90.0,
                _ => 1.0,
            };
            events.summary(epoch * 7, &[("loss".into(), loss)], &[])?;
        }
        events.summary(192 * 7, &[("loss".into(), 2.0)], &[])?;
        for epoch in 193..=10_200 {
            let loss = if epoch == 5017 { 7.0 } else { 1.0 };
            events.summary(epoch * 7, &[("loss".into(), loss)], &[])?;
        }
        let series = scalars(directory.path())?;
        let points = &series["loss"];
        assert!(points.len() <= 600);
        assert_eq!(points.front(), Some(&(7, 1.0)));
        assert_eq!(points.back(), Some(&(10_200 * 7, 1.0)));
        assert_eq!(
            points.iter().map(|point| point.1).reduce(f32::min),
            Some(-9.0)
        );
        assert_eq!(
            points.iter().map(|point| point.1).reduce(f32::max),
            Some(7.0)
        );
        assert!(
            points
                .iter()
                .zip(points.iter().skip(1))
                .all(|(a, b)| a.0 < b.0)
        );
        Ok(())
    }

    #[test]
    fn reads_scalars_while_the_next_record_is_being_written() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut events = Events::new(directory.path())?;
        events.summary(10, &[("loss/g/total".into(), 1.5)], &[])?;
        events.0.write_all(&[12, 0, 0])?;
        events.0.flush()?;
        let values = scalars(directory.path())?;
        assert_eq!(values["loss/g/total"].as_slices().0, &[(10, 1.5)]);
        Ok(())
    }
}
