//! Channel tables from atlas label files. Several formats, because atlases name and describe
//! their regions differently (docs/regions.md).

use std::path::Path;

use modewise_brain::{ChannelInfo, ChannelTable, Hemisphere, Region};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableFormat {
    /// Tab-separated with a header: `name` (required), `region`, `hemisphere`; `index` and
    /// `color` are ignored; every other column becomes a tag. Covers BIDS `dseg.tsv`.
    Tsv,
    /// Schaefer or Yan (homotopic) cortical labels such as `17Networks_LH_DefaultB_PFCm_1`, one
    /// per line or as a column of a lookup table: hemisphere, `yeo7`/`yeo17` network tag, region.
    Schaefer,
    /// Tian subcortical labels such as `lAMY-rh`, one per line: region and hemisphere.
    Tian,
    /// One name per line, with no further description.
    Names,
}

pub fn read_table(path: &Path, format: TableFormat) -> Result<ChannelTable, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let channels: Vec<ChannelInfo> = match format {
        TableFormat::Tsv => {
            return parse_tsv(&text)
                .and_then(|c| ChannelTable::new(c).map_err(|e| e.to_string()))
                .map_err(|e| format!("{}: {e}", path.display()));
        }
        TableFormat::Schaefer => lines
            .filter_map(|l| l.split_whitespace().find_map(schaefer_channel))
            .collect(),
        TableFormat::Tian => lines.map(tian_channel).collect(),
        TableFormat::Names => lines.map(ChannelInfo::new).collect(),
    };
    if channels.is_empty() {
        return Err(format!("{}: no channel labels found", path.display()));
    }
    ChannelTable::new(channels).map_err(|e| format!("{}: {e}", path.display()))
}

/// `<k>Networks_<LH|RH>_<network>[_<region>…][_<n>]`, as in the Schaefer and Yan parcellations.
/// Returns `None` for text that is not such a label.
pub fn schaefer_channel(label: &str) -> Option<ChannelInfo> {
    let mut parts = label.split('_');
    let prefix = parts.next()?;
    let networks = prefix
        .strip_suffix("Networks")
        .or_else(|| prefix.strip_suffix("networks"))?;
    if networks.is_empty() || !networks.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let hemisphere = Hemisphere::parse(parts.next()?)?;
    let network = parts.next()?;
    let mut rest: Vec<&str> = parts.collect();
    if rest
        .last()
        .is_some_and(|p| p.bytes().all(|b| b.is_ascii_digit()))
    {
        rest.pop();
    }
    let channel = ChannelInfo::new(label)
        .with_hemisphere(hemisphere)
        .with_tag(format!("yeo{networks}"), network);
    Some(if rest.is_empty() {
        channel
    } else {
        channel.with_region(Region::new(rest.join("_")))
    })
}

/// `<region>-<lh|rh>`, as in the Tian subcortex atlas; without a hemisphere suffix the whole
/// label is the region.
pub fn tian_channel(label: &str) -> ChannelInfo {
    match label
        .rsplit_once('-')
        .and_then(|(region, side)| Some((region, Hemisphere::parse(side)?)))
    {
        Some((region, side)) => ChannelInfo::new(label)
            .with_region(Region::new(region))
            .with_hemisphere(side),
        None => ChannelInfo::new(label).with_region(Region::new(label)),
    }
}

fn parse_tsv(text: &str) -> Result<Vec<ChannelInfo>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or("empty file")?
        .split('\t')
        .map(str::trim)
        .collect();
    let column = |name: &str| header.iter().position(|h| *h == name);
    let name = column("name").ok_or("no `name` column")?;
    let (region, hemisphere) = (column("region"), column("hemisphere"));
    lines
        .enumerate()
        .map(|(row, line)| {
            let cells: Vec<&str> = line.split('\t').map(str::trim).collect();
            let cell = |i: usize| {
                cells
                    .get(i)
                    .copied()
                    .filter(|v| !v.is_empty() && *v != "n/a")
            };
            let mut channel =
                ChannelInfo::new(cell(name).ok_or(format!("row {}: empty name", row + 2))?);
            if let Some(r) = region.and_then(cell) {
                channel = channel.with_region(Region::new(r));
            }
            if let Some(h) = hemisphere.and_then(cell) {
                let side = Hemisphere::parse(h).ok_or(format!(
                    "row {}: hemisphere {h:?} is not left or right",
                    row + 2
                ))?;
                channel = channel.with_hemisphere(side);
            }
            for (i, key) in header.iter().enumerate() {
                let described = [Some(name), region, hemisphere].contains(&Some(i));
                if !described
                    && !["index", "color"].contains(key)
                    && let Some(value) = cell(i)
                {
                    channel = channel.with_tag(*key, value);
                }
            }
            Ok(channel)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schaefer_labels_give_hemisphere_network_and_region() {
        let c = schaefer_channel("17networks_LH_DefaultB_PFCm_1").unwrap();
        assert_eq!(
            (
                c.hemisphere(),
                c.tag("yeo17"),
                c.region().map(Region::as_str)
            ),
            (Some(Hemisphere::Left), Some("DefaultB"), Some("PFCm"))
        );
        let striate = schaefer_channel("17networks_RH_VisCent_Striate").unwrap();
        assert_eq!(
            striate.region().map(Region::as_str),
            Some("Striate"),
            "no trailing index"
        );
        let sommot = schaefer_channel("17networks_LH_SomMotB_7").unwrap();
        assert_eq!(
            (sommot.region(), sommot.tag("yeo17")),
            (None, Some("SomMotB")),
            "no region, only a network"
        );
        let seven = schaefer_channel("7Networks_RH_Default_pCunPCC_3").unwrap();
        assert_eq!(
            (seven.tag("yeo7"), seven.hemisphere()),
            (Some("Default"), Some(Hemisphere::Right))
        );
        let two_part = schaefer_channel("17Networks_LH_ContA_PFCl_post_2").unwrap();
        assert_eq!(two_part.region().map(Region::as_str), Some("PFCl_post"));
        for not_a_label in [
            "aHIP-rh",
            "Networks_LH_X_1",
            "17Networks_XH_X_1",
            "17Networks_LH",
            "1 12 48 255 255",
        ] {
            assert!(schaefer_channel(not_a_label).is_none(), "{not_a_label}");
        }
    }

    #[test]
    fn tian_labels_give_region_and_hemisphere() {
        let c = tian_channel("NAc-shell-rh");
        assert_eq!(
            (c.region().map(Region::as_str), c.hemisphere()),
            (Some("NAc-shell"), Some(Hemisphere::Right))
        );
        let c = tian_channel("lAMY-lh");
        assert_eq!(
            (c.region().map(Region::as_str), c.hemisphere()),
            (Some("lAMY"), Some(Hemisphere::Left))
        );
        let c = tian_channel("brainstem");
        assert_eq!(
            (c.region().map(Region::as_str), c.hemisphere()),
            (Some("brainstem"), None)
        );
    }

    #[test]
    fn tsv_tables_read_columns_and_tags() {
        let text = "index\tname\tregion\themisphere\tcolor\tlobe\n1\tFp1\tfrontopolar\tL\t#fff\tfrontal\n2\tCz\tcentral\tn/a\t#000\t\n";
        let channels = parse_tsv(text).unwrap();
        assert_eq!(channels[0].tag("lobe"), Some("frontal"));
        assert_eq!(channels[0].tag("color"), None);
        assert_eq!(channels[0].hemisphere(), Some(Hemisphere::Left));
        assert_eq!(
            (channels[1].hemisphere(), channels[1].tag("lobe")),
            (None, None)
        );
        assert!(parse_tsv("label\n").unwrap_err().contains("`name`"));
        assert!(
            parse_tsv("name\themisphere\nx\tup\n")
                .unwrap_err()
                .contains("row 2")
        );
        assert!(
            parse_tsv("name\tregion\n\tPFC\n")
                .unwrap_err()
                .contains("empty name")
        );
        assert!(parse_tsv("").is_err());
    }
}
