//! Tonal ramp remapping: the palette is read as a neutral ramp plus a set of accents.
//!
//! Instead of interpolating between the nearest palette colors (which sends most midtones and
//! shadows to the background shades), each color keeps its place in the image's tonal structure:
//!
//! 1. The palette is split into neutrals (a ramp from the darkest to the lightest gray-ish color)
//!    and accents, automatically, without named roles.
//! 2. Lightness: the input's Oklab L is spread over the neutral ramp, then blended with the
//!    original lightness (`keep_lum`).
//! 3. Hue: the input's a/b goes to a Gaussian-weighted mix of the nearest accents (`smooth`),
//!    renormalized to the accents' chroma, and applied in proportion to the input's chroma: grays
//!    stay on the neutral ramp, saturated colors become accents. Accent chroma is reduced on dark
//!    pixels to avoid neon shadows.

use std::f64::consts::{PI, TAU};

use arrayref::array_ref;
use oklab::{oklab_to_srgb, srgb_to_oklab, Oklab};

use super::InterpolatedRemapper;

/// Max distance in the a/b plane between a neutral and the ramp through the other neutrals.
const RAMP_TOLERANCE: f64 = 0.02;
/// Number of least chromatic colors whose pairs are tried as ramp seeds.
const SEEDS_FROM: usize = 8;
/// A ramp end this far (in L) from the palette's darkest/lightest color is extended to it.
const END_GAP: f64 = 0.10;
/// ... with that extreme color if it is this unsaturated, else with a desaturated copy.
const END_MAX_CHROMA: f64 = 0.06;
/// Input chroma at which a color fully takes its accent.
const CHROMA_REF: f64 = 0.10;
/// Hue gaps between neighboring accents wider than this cap the chroma renormalization, so two
/// nearly opposite accents meet through a gray instead of flipping hue abruptly.
const GAP_REF: f64 = 75.0 * PI / 180.0;
/// Max chroma renormalization in the middle of a wide hue gap.
const MAX_RESTORE: f64 = 1.5;

/// Remapper that keeps the image's tonal structure: lightness follows the palette's neutral
/// ramp, hue goes to the nearest accents in proportion to the original chroma.
pub struct TonalRampRemapper {
    /// Neutral ramp, sorted by lightness.
    neutrals: Vec<[f64; 3]>,
    /// The same, one channel per vector, for interpolation.
    ramp: [Vec<f64>; 3],
    /// Accents in Oklab, with their chroma.
    accents: Vec<([f64; 3], f64)>,
    /// Accent hues (radians, 0..2pi), sorted, and the gap following each of them.
    hues: Vec<(f64, f64)>,
    mean_accent_l: f64,
    keep_lum: f64,
    smooth: f64,
}

impl TonalRampRemapper {
    /// Create a new remapper.
    ///
    /// - `keep_lum`: 0 = the palette's lightness, 1 = the original lightness (default 0.5).
    /// - `smooth`: blend radius between accents in the Oklab a/b plane; smaller = purer palette
    ///   colors (default 0.08).
    pub fn new(palette: &[[u8; 3]], keep_lum: f64, smooth: f64) -> Self {
        let lab: Vec<[f64; 3]> = palette.iter().map(|&c| to_lab(c)).collect();
        let (mut neutrals, accent_idx) = split(&lab);
        neutrals.sort_by(|a, b| a[0].total_cmp(&b[0]));

        let accents: Vec<_> = accent_idx
            .iter()
            .map(|&i| (lab[i], chroma(&lab[i])))
            .collect();
        let mut sorted: Vec<f64> = accents
            .iter()
            .map(|(c, _)| c[2].atan2(c[1]).rem_euclid(TAU))
            .collect();
        sorted.sort_by(f64::total_cmp);
        let hues = (0..sorted.len())
            .map(|i| {
                let next = sorted.get(i + 1).copied().unwrap_or(sorted[0] + TAU);
                (sorted[i], next - sorted[i])
            })
            .collect();
        let mean_accent_l = if accents.is_empty() {
            1.0
        } else {
            accents.iter().map(|(c, _)| c[0]).sum::<f64>() / accents.len() as f64
        };

        let ramp = std::array::from_fn(|k| neutrals.iter().map(|c| c[k]).collect());

        Self {
            neutrals,
            ramp,
            accents,
            hues,
            mean_accent_l,
            keep_lum,
            smooth,
        }
    }

    /// The neutral ramp found in the palette (sorted by lightness) and the accents, as sRGB.
    pub fn roles(&self) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let rgb = |c: &[f64; 3]| {
            oklab_to_srgb(Oklab {
                l: c[0] as f32,
                a: c[1] as f32,
                b: c[2] as f32,
            })
            .into()
        };
        (
            self.neutrals.iter().map(rgb).collect(),
            self.accents.iter().map(|(c, _)| rgb(c)).collect(),
        )
    }

    /// Neutral ramp color at a lightness, interpolated (clamped at both ends).
    fn neutral_at(&self, l: f64) -> [f64; 3] {
        std::array::from_fn(|k| interp(l, &self.ramp[0], &self.ramp[k]))
    }

    /// Cap on the chroma renormalization for an input hue: none next to the accents, up to
    /// [`MAX_RESTORE`] in the middle of a hue gap wider than [`GAP_REF`].
    fn restore_cap(&self, hue: f64) -> f64 {
        if self.hues.len() < 2 {
            return f64::INFINITY;
        }
        let h = hue.rem_euclid(TAU);
        let idx = match self.hues.partition_point(|&(s, _)| s <= h) {
            0 => self.hues.len() - 1,
            n => n - 1,
        };
        let (start, gap) = self.hues[idx];
        if gap <= GAP_REF {
            return f64::INFINITY;
        }
        let pos = (h - start).rem_euclid(TAU) / gap;
        MAX_RESTORE / (PI * pos).sin().max(1e-6)
    }
}

impl<'a> InterpolatedRemapper<'a> for TonalRampRemapper {
    fn remap_pixel(&self, pixel: &mut image::Rgba<u8>) {
        let [l, a, b] = to_lab(*array_ref![pixel.0, 0, 3]);

        // lightness: spread over the neutral ramp, blended with the original
        let lo = self.neutrals[0][0];
        let hi = self.neutrals[self.neutrals.len() - 1][0];
        let neutral = self.neutral_at(lo + l.clamp(0.0, 1.0) * (hi - lo));
        let out_l = self.keep_lum * l + (1.0 - self.keep_lum) * neutral[0];

        let (out_a, out_b) = if self.accents.is_empty() {
            (neutral[1], neutral[2])
        } else {
            // accent mix by distance in the a/b plane
            let d2 = |c: &[f64; 3]| (a - c[1]).powi(2) + (b - c[2]).powi(2);
            let min = self
                .accents
                .iter()
                .map(|(c, _)| d2(c))
                .fold(f64::INFINITY, f64::min);
            let (mut wsum, mut ma, mut mb, mut mc) = (0.0, 0.0, 0.0, 0.0);
            for (c, ch) in &self.accents {
                let w = (-(d2(c) - min) / (2.0 * self.smooth * self.smooth)).exp();
                wsum += w;
                ma += w * c[1];
                mb += w * c[2];
                mc += w * ch;
            }
            let (ma, mb, mc) = (ma / wsum, mb / wsum, mc / wsum);
            // mixing neighboring accents shortens the a/b vector: restore the accents' chroma
            let f = (mc / ma.hypot(mb).max(1e-6)).min(self.restore_cap(b.atan2(a)));
            let (acc_a, acc_b) = (ma * f, mb * f);

            let k = (a.hypot(b) / CHROMA_REF).clamp(0.0, 1.0);
            // darker than the accents -> proportionally less chroma
            let dim = (out_l / (0.75 * self.mean_accent_l)).clamp(0.0, 1.0);
            (
                (1.0 - k) * neutral[1] + k * acc_a * dim,
                (1.0 - k) * neutral[2] + k * acc_b * dim,
            )
        };

        let rgb = oklab_to_srgb(Oklab {
            l: out_l as f32,
            a: out_a as f32,
            b: out_b as f32,
        });
        pixel.0[0..3].copy_from_slice(rgb.as_ref());
    }
}

fn to_lab(c: [u8; 3]) -> [f64; 3] {
    let Oklab { l, a, b } = srgb_to_oklab(c.into());
    [l as f64, a as f64, b as f64]
}

fn chroma(c: &[f64; 3]) -> f64 {
    c[1].hypot(c[2])
}

/// Piecewise linear interpolation, clamped outside `xp` (like numpy's `interp`). `xp` is sorted.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    let t = (x - xp[j]) / (xp[j + 1] - xp[j]);
    fp[j] + t * (fp[j + 1] - fp[j])
}

/// Distance in the a/b plane between color `i` and the ramp through `members` (minus `i`).
fn ramp_dev(lab: &[[f64; 3]], members: &[usize], i: usize) -> f64 {
    let mut m: Vec<usize> = members.iter().copied().filter(|&j| j != i).collect();
    match m.len() {
        0 => return f64::INFINITY,
        1 => return (lab[i][1] - lab[m[0]][1]).hypot(lab[i][2] - lab[m[0]][2]),
        _ => {},
    }
    m.sort_by(|&x, &y| lab[x][0].total_cmp(&lab[y][0]));
    let ls: Vec<f64> = m.iter().map(|&j| lab[j][0]).collect();
    let pa: Vec<f64> = m.iter().map(|&j| lab[j][1]).collect();
    let pb: Vec<f64> = m.iter().map(|&j| lab[j][2]).collect();
    let (ea, eb) = (interp(lab[i][0], &ls, &pa), interp(lab[i][0], &ls, &pb));
    (lab[i][1] - ea).hypot(lab[i][2] - eb)
}

/// Drop the worst outlier of the ramp while one is off by `RAMP_TOLERANCE` or more, else add
/// the closest outsider while it is within it; until nothing changes.
fn refine(lab: &[[f64; 3]], seed: &[usize]) -> Vec<usize> {
    let mut members: Vec<usize> = seed.to_vec();
    members.sort_unstable();
    for _ in 0..4 * lab.len() {
        if members.len() > 2 {
            let (worst, dev) = members
                .iter()
                .map(|&i| (i, ramp_dev(lab, &members, i)))
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            if dev >= RAMP_TOLERANCE {
                members.retain(|&i| i != worst);
                continue;
            }
        }
        let best = (0..lab.len())
            .filter(|i| !members.contains(i))
            .map(|i| (i, ramp_dev(lab, &members, i)))
            .fold(None, |a: Option<(usize, f64)>, b| match a {
                Some(a) if a.1 <= b.1 => Some(a),
                _ => Some(b),
            });
        match best {
            Some((i, dev)) if dev < RAMP_TOLERANCE => {
                members.push(i);
                members.sort_unstable();
            },
            _ => break,
        }
    }
    members
}

/// Split a palette into its neutral ramp (Oklab colors, possibly with synthesized ends) and the
/// indices of its accents.
///
/// A color is a neutral if its a/b is within `RAMP_TOLERANCE` of the ramp interpolated through
/// the other neutrals at its lightness. The largest ramp is kept among those grown from several
/// seeds: the less chromatic half of the palette, and every pair among its least chromatic
/// colors. Ties go to the ramp with the lower mean chroma.
fn split(lab: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<usize>) {
    let n = lab.len();
    let c: Vec<f64> = lab.iter().map(chroma).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| c[x].total_cmp(&c[y]));

    let mut seeds = vec![order[..(n / 2).max(2).min(n)].to_vec()];
    let low = &order[..SEEDS_FROM.min(n)];
    for (i, &x) in low.iter().enumerate() {
        for &y in &low[i + 1..] {
            seeds.push(vec![x, y]);
        }
    }
    let mean_c = |m: &[usize]| m.iter().map(|&i| c[i]).sum::<f64>() / m.len() as f64;
    let mut members =
        seeds
            .iter()
            .map(|s| refine(lab, s))
            .fold(Vec::new(), |best: Vec<usize>, m| {
                let better =
                    m.len() > best.len() || (m.len() == best.len() && mean_c(&m) < mean_c(&best));
                if better {
                    m
                } else {
                    best
                }
            });
    if members.len() == 2 && ramp_dev(lab, &members, members[0]) >= RAMP_TOLERANCE {
        members = vec![order[0]]; // no ramp at all: the least chromatic color
    }

    // the ramp must span the palette's lightness, or every image is pushed light or dark
    let mut ramp: Vec<usize> = members.clone();
    ramp.sort_by(|&x, &y| lab[x][0].total_cmp(&lab[y][0]));
    let by_l = |x: &usize, y: &usize| lab[*x][0].total_cmp(&lab[*y][0]);
    let darkest = (0..n).min_by(by_l).unwrap();
    let lightest = (0..n).max_by(by_l).unwrap();
    let mut synth = Vec::new();
    for (end, extreme) in [(ramp[0], darkest), (ramp[ramp.len() - 1], lightest)] {
        if (lab[extreme][0] - lab[end][0]).abs() <= END_GAP || members.contains(&extreme) {
            continue;
        }
        if c[extreme] < END_MAX_CHROMA {
            members.push(extreme);
        } else {
            synth.push([lab[extreme][0], lab[end][1], lab[end][2]]);
        }
    }

    let neutrals = members.iter().map(|&i| lab[i]).chain(synth).collect();
    let accents = (0..n).filter(|i| !members.contains(i)).collect();
    (neutrals, accents)
}

#[cfg(test)]
mod tests {
    use lutgen_palettes::Palette;

    use super::*;
    use crate::GenerateLut;

    fn hex(c: [u8; 3]) -> String {
        format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
    }

    #[test]
    fn catppuccin_roles_are_recovered() {
        // every flavour: the 12 neutrals (crust..text) and the 14 accents, without named roles
        for palette in [
            Palette::CatppuccinLatte,
            Palette::CatppuccinFrappe,
            Palette::CatppuccinMacchiato,
            Palette::CatppuccinMocha,
        ] {
            let colors = palette.get();
            let lab: Vec<_> = colors.iter().map(|&c| to_lab(c)).collect();
            let (neutrals, accents) = split(&lab);
            assert_eq!(neutrals.len(), 12, "{palette:?}");
            assert_eq!(accents.len(), 14, "{palette:?}");
        }
        // Mocha's rosewater and flamingo are less chromatic than its text: still accents
        let mocha = Palette::CatppuccinMocha.get();
        let (_, accents) = TonalRampRemapper::new(mocha, 0.5, 0.08).roles();
        let accents: Vec<_> = accents.into_iter().map(hex).collect();
        assert!(accents.contains(&"#f5e0dc".to_string()), "{accents:?}");
        assert!(accents.contains(&"#f2cdcd".to_string()), "{accents:?}");
    }

    #[test]
    fn grays_stay_neutral() {
        let remapper = TonalRampRemapper::new(Palette::GruvboxDark.get(), 0.5, 0.08);
        for v in [0u8, 64, 128, 192, 255] {
            let mut px = image::Rgba([v, v, v, 255]);
            remapper.remap_pixel(&mut px);
            let lab = to_lab(*array_ref![px.0, 0, 3]);
            assert!(chroma(&lab) < 0.06, "gray {v} -> {:?}", px.0);
        }
    }

    #[test]
    fn degenerate_palettes() {
        // monochrome (no accents), a single color, and two colors
        for colors in [
            vec![[0, 0, 0], [128, 128, 128], [255, 255, 255]],
            vec![[200, 30, 30]],
            vec![[0, 0, 0], [0, 200, 0]],
        ] {
            let lut = TonalRampRemapper::new(&colors, 0.5, 0.08).generate_lut(4);
            assert_eq!(lut.width(), 64);
        }
    }

    #[test]
    fn no_hue_flip_in_wide_gaps() {
        // Gruvbox has no accent between its blue (~199 deg) and its pink (~352 deg): walking
        // around the hue circle, the output must change smoothly.
        let remapper = TonalRampRemapper::new(Palette::GruvboxDark.get(), 0.5, 0.08);
        let out: Vec<[f64; 3]> = (0..360)
            .step_by(3)
            .map(|h| {
                let h = (h as f64).to_radians();
                let rgb = oklab_to_srgb(Oklab {
                    l: 0.65,
                    a: (0.13 * h.cos()) as f32,
                    b: (0.13 * h.sin()) as f32,
                });
                let mut px = image::Rgba([rgb.r, rgb.g, rgb.b, 255]);
                remapper.remap_pixel(&mut px);
                to_lab(*array_ref![px.0, 0, 3])
            })
            .collect();
        for (i, c) in out.iter().enumerate() {
            let d = out[(i + 1) % out.len()];
            let step = (c[1] - d[1]).hypot(c[2] - d[2]);
            assert!(step < 0.03, "a/b step {step:.3} at {} deg", i * 3);
        }
    }
}
