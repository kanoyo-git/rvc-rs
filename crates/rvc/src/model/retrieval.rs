use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

pub struct Index {
    dimensions: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    lists: Vec<Vec<f32>>,
}

impl Index {
    pub fn load(path: &Path, dimensions: usize) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let size = file.metadata()?.len();
        ensure!(size <= 2 * 1024 * 1024 * 1024, "FAISS file exceeds 2 GiB");
        let mut r = Reader {
            data: BufReader::new(file),
            remaining: size as usize,
        };
        let code: [u8; 4] = r.data()?;
        let (d, count) = r.header()?;
        ensure!(
            d == dimensions && count > 0 && count <= r.remaining / (d * 4),
            "unexpected FAISS dimensions or vector count"
        );
        let (nprobe, centroids, lists) = match &code {
            b"IxF2" | b"IxFl" => {
                let vectors = r.flat(d, count)?;
                (1, vec![0.0; d], vec![vectors])
            }
            b"IwFl" => {
                let nlist = r.size()?;
                let nprobe = r.size()?;
                ensure!(
                    (1..=100_000).contains(&nlist) && (1..=nlist).contains(&nprobe),
                    "invalid FAISS list count"
                );
                let tag: [u8; 4] = r.data()?;
                let (cd, nc) = r.header()?;
                ensure!(
                    matches!(&tag, b"IxF2" | b"IxFl") && cd == d && nc == nlist,
                    "expected Flat-L2 IVF centroids"
                );
                let centroids = r.flat(d, nlist)?;
                let [map]: [u8; 1] = r.data()?;
                ensure!(map <= 2, "invalid FAISS direct map");
                let size = r.size()?;
                r.skip_elements(size, 8)?;
                if map == 2 {
                    let size = r.size()?;
                    r.skip_elements(size, 16)?;
                }
                ensure!(
                    r.data::<4>()? == *b"ilar",
                    "FAISS index must have embedded ArrayInvertedLists"
                );
                ensure!(
                    r.size()? == nlist && r.size()? == d * 4,
                    "invalid FAISS list geometry"
                );
                let encoding = r.data::<4>()?;
                let length = r.size()?;
                let mut sizes = vec![0usize; nlist];
                match &encoding {
                    b"full" if length == nlist => {
                        for size in &mut sizes {
                            *size = r.size()?;
                        }
                    }
                    b"sprs" if length % 2 == 0 && length / 2 <= nlist => {
                        for _ in 0..length / 2 {
                            let list = r.size()?;
                            let size = r.size()?;
                            let target = sizes
                                .get_mut(list)
                                .context("invalid sparse FAISS list ID")?;
                            ensure!(
                                *target == 0 && size > 0,
                                "duplicate or empty sparse FAISS list"
                            );
                            *target = size;
                        }
                    }
                    _ => bail!("invalid FAISS list length encoding"),
                }
                let reported = sizes
                    .iter()
                    .try_fold(0usize, |sum, &size| sum.checked_add(size))
                    .context("FAISS vector count overflow")?;
                ensure!(
                    reported == count && count <= r.remaining / (d * 4 + 8),
                    "invalid number of stored FAISS vectors"
                );
                let mut seen = vec![false; count];
                let mut lists = Vec::with_capacity(nlist);
                for size in sizes {
                    lists.push(r.floats(size * d)?);
                    for _ in 0..size {
                        let id = r.size()?;
                        let seen = seen
                            .get_mut(id)
                            .context("FAISS vector ID is outside the stored range")?;
                        ensure!(!*seen, "duplicate FAISS vector ID");
                        *seen = true;
                    }
                }
                (nprobe, centroids, lists)
            }
            _ => bail!("unsupported FAISS index type; expected Flat-L2 or IVF-Flat-L2"),
        };
        Ok(Self {
            dimensions,
            nprobe,
            centroids,
            lists,
        })
    }

    pub fn mix(&self, features: &mut [f32], rate: f32) {
        features
            .par_chunks_exact_mut(self.dimensions)
            .for_each(|row| self.mix_frame(row, rate));
    }

    fn mix_frame(&self, row: &mut [f32], ratio: f32) {
        let distance =
            |v: &[f32]| -> f32 { row.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum() };
        let mut probes: Vec<_> = self
            .centroids
            .chunks_exact(self.dimensions)
            .enumerate()
            .map(|(i, v)| (distance(v), i))
            .collect();
        if self.nprobe < probes.len() {
            probes.select_nth_unstable_by(self.nprobe, |a, b| {
                a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
            });
        }
        let mut best: [(f32, &[f32]); 8] = [(f32::INFINITY, &[]); 8];
        for &(_, list) in &probes[..self.nprobe] {
            for v in self.lists[list].chunks_exact(self.dimensions) {
                let d = distance(v);
                if d < best[7].0 {
                    let place = best.partition_point(|entry| entry.0 <= d);
                    best.copy_within(place..7, place + 1);
                    best[place] = (d, v);
                }
            }
        }
        let neighbors: Vec<_> = best
            .into_iter()
            .take_while(|(_, v)| !v.is_empty())
            .map(|(d, v)| (f64::from(d.max(1e-12)).powi(-2), v))
            .collect();
        let sum: f64 = neighbors.iter().map(|(w, _)| w).sum();
        if sum > 0.0 {
            for (i, value) in row.iter_mut().enumerate() {
                let feature: f64 = neighbors.iter().map(|(w, v)| w * f64::from(v[i])).sum();
                *value = *value * (1.0 - ratio) + (feature / sum) as f32 * ratio;
            }
        }
    }
}

struct Reader<R> {
    data: R,
    remaining: usize,
}

impl<R: Read> Reader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes.len())
            .context("truncated FAISS index")?;
        self.data.read_exact(bytes)?;
        Ok(())
    }

    fn data<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read(&mut bytes)?;
        Ok(bytes)
    }

    fn size(&mut self) -> Result<usize> {
        Ok(u64::from_le_bytes(self.data()?).try_into()?)
    }

    fn header(&mut self) -> Result<(usize, usize)> {
        let d: usize = i32::from_le_bytes(self.data()?).try_into()?;
        let count: usize = i64::from_le_bytes(self.data()?).try_into()?;
        let _reserved: [u8; 16] = self.data()?;
        ensure!(self.data::<1>()? == [1], "FAISS index is untrained");
        ensure!(
            i32::from_le_bytes(self.data()?) == 1,
            "FAISS index must use L2 distance"
        );
        Ok((d, count))
    }

    fn skip_elements(&mut self, elements: usize, element_size: usize) -> Result<()> {
        let mut count = elements
            .checked_mul(element_size)
            .filter(|n| *n <= self.remaining)
            .context("invalid FAISS direct map")?;
        let mut buffer = [0; 4096];
        while count != 0 {
            let size = count.min(buffer.len());
            self.read(&mut buffer[..size])?;
            count -= size;
        }
        Ok(())
    }

    fn flat(&mut self, d: usize, count: usize) -> Result<Vec<f32>> {
        let elements = d.checked_mul(count).context("FAISS dimensions overflow")?;
        ensure!(
            self.size()? == elements,
            "invalid size of FAISS Flat-L2 vectors"
        );
        self.floats(elements)
    }

    fn floats(&mut self, elements: usize) -> Result<Vec<f32>> {
        ensure!(elements <= self.remaining / 4, "truncated FAISS vectors");
        let mut result = Vec::with_capacity(elements);
        let mut buffer = [0; 8192];
        while result.len() < elements {
            let size = (elements - result.len()).min(buffer.len() / 4) * 4;
            self.read(&mut buffer[..size])?;
            for &chunk in buffer[..size].as_chunks::<4>().0 {
                let number = f32::from_le_bytes(chunk);
                ensure!(number.is_finite(), "FAISS index has non-finite values");
                result.push(number);
            }
        }
        Ok(result)
    }
}
