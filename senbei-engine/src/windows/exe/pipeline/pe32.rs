use super::super::super::layout;
use super::*;

fn stage_key_rounds(data: &[u8], table: u32, slots: usize) -> Result<u32, UnpackError> {
    let offset = table as usize;
    let size = slots.saturating_mul(16);
    let descriptors = offset
        .checked_add(size)
        .and_then(|end| data.get(offset..end))
        .ok_or(UnpackError::BufferRangeOutOfBounds {
            operation: BufferOperation::Read,
            offset,
            size,
            buffer_len: data.len(),
        })?;
    // The loader stops at the first empty helper, even if later slots are nonempty.
    Ok(descriptors
        .as_chunks::<16>()
        .0
        .iter()
        .take_while(|descriptor| get_u32(descriptor.as_slice(), 4) > 4)
        .count() as u32)
}

impl<'a> Unpacker<'a> {
    /// PE32 (32-bit) unpack pipeline. The shared Stage 1/2 setup (info decrypt,
    /// payload decrypt, raw copy, header restore) has already run in `run()`
    /// before dispatch; this takes over from "Locating shell offsets".
    pub(super) fn run_pe32(&mut self, pe_off: u32, verbose: bool) -> Result<Vec<u8>, UnpackError> {
        let info = self.info;
        let info3 = info[3];

        // advance_key: replays the packer's per-iteration key walk.
        let advance_key = |mut key: u32, iterations: u32| -> u32 {
            for m in 0..iterations {
                let bound = (m + 1).wrapping_mul(25) << 2;
                let mut n: u32 = 1;
                while n <= bound {
                    key = key.wrapping_add(n);
                    n += 1;
                }
            }
            key
        };

        // ---- Locate tbl in shell ----
        let tbl =
            layout::find_tbl_pe32(&self.decompressed, &info).ok_or(UnpackError::Pe32TblNotFound)?;
        if verbose {
            println!("[3/9] Locating config layout (PE32)...");
            println!("  tbl = 0x{:X}", tbl);
        }

        // Earlier PE32 layouts terminate the checksum chain at `tbl+0x58`
        // immediately and keep the active chain at `tbl+0x60`.
        let old_layout = get_u32(&self.decompressed, tbl.wrapping_add(0x5C)) == 0;
        if old_layout {
            let payload_lo = info3 as usize;
            let payload_hi = payload_lo
                .saturating_add(self.decrypt_size.max(info[5]) as usize)
                .min(self.decompressed.len());
            self.map_raw_sections(pe_off, payload_lo, payload_hi);
        }

        // ---- PE header restore ----
        let val_bc = get_u32(&self.decompressed, tbl.wrapping_add(0xBC));
        let (val_88, val_8c) = if old_layout {
            (
                get_u32(&self.decompressed, tbl.wrapping_add(0xC0)),
                get_u32(&self.decompressed, tbl.wrapping_add(0xC4)),
            )
        } else {
            (
                get_u32(&self.decompressed, tbl.wrapping_add(0xC8)),
                get_u32(&self.decompressed, tbl.wrapping_add(0xCC)),
            )
        };
        let image_len = self.decompressed.len() as u64;
        let plausible = |value: u32| (value as u64) < image_len;
        if plausible(val_bc) && plausible(val_88) && plausible(val_8c) {
            write_u32(&mut self.decompressed, pe_off.wrapping_add(0x80), val_bc);
            write_u32(&mut self.decompressed, pe_off.wrapping_add(0x88), val_88);
            write_u32(&mut self.decompressed, pe_off.wrapping_add(0x8C), val_8c);
            if old_layout {
                write_u32(&mut self.decompressed, pe_off.wrapping_add(0xA0), 0);
                write_u32(&mut self.decompressed, pe_off.wrapping_add(0xA4), 0);
            } else {
                write_u32(&mut self.decompressed, pe_off.wrapping_add(0xB0), 0);
                write_u32(&mut self.decompressed, pe_off.wrapping_add(0xB4), 0);
            }
        }

        // ---- Header-independent checksum inputs ----
        let first_stage_cs = self.calculate_checksum(tbl.wrapping_add(0xA8));
        let second_stage_key_offsets: &[u32] = if old_layout { &[0x48, 0x50] } else { &[0x40] };

        // ---- Stage 3: SecondStage ----
        //
        // ss_key = headerChecksum ^ firstStageCS ^ secondStageKey, where the
        // header checksum (a XOR of crc32(region)^size over the sub-regions at
        // tbl+0x58) is taken over the *original* pre-pack PE header. For EXEs the
        // import/resource restore above reconstructs that header exactly. Native
        // DLLs additionally carry a packer-added BaseReloc data-directory entry
        // (dir 5) that was absent from the checksummed original, so the header
        // checksum only matches once that entry is treated as zero. EXEs have no
        // dir-5 entry, so zeroing it is a no-op for them.
        //
        // Rather than branch on EXE-vs-DLL, try the header as-is and, on failure,
        // with the BaseReloc entry zeroed; keep whichever ss_key decrypts a
        // SecondStage whose ThirdStage (off,size) pair lands inside the image.
        // This uses the same shift/key trial-and-validate the later stages
        // already use, and keeps EXE output byte-identical (the as-is variant
        // wins first).
        let ss_pair = tbl.wrapping_add(0x98);
        let ss = get_u32(&self.decompressed, ss_pair);
        let ss_size = get_u32(&self.decompressed, ss_pair.wrapping_add(4));
        let ss_shift = ss_size.wrapping_sub(0xBC0);
        // Back up the SecondStage ciphertext so a failed trial can be retried.
        let ss_lo = ss as usize;
        let ss_hi = ss_lo.wrapping_add(ss_size as usize);
        if ss_hi < ss_lo || ss_hi > self.decompressed.len() {
            return Err(UnpackError::Pe32SecondStageRangeInvalid {
                offset: ss,
                size: ss_size,
                image_len: self.decompressed.len(),
            });
        }
        let ss_ct: Vec<u8> = self.decompressed[ss_lo..ss_hi].to_vec();
        // PE32 data dir 5 (BaseReloc) = optional_header(pe+24) + 0x60 + 5*8 = pe+0xA0.
        let reloc_dir = pe_off.wrapping_add(0xA0);
        let len = self.decompressed.len() as u64;
        let pair_off = 0xB8Cu32.wrapping_add(ss_shift);
        let mut found = false;
        // Holds the winning variant's header checksum; the later stages
        // (Forth/Fifth/Seven/Eighth) reuse it as a key component.
        let mut header_checksum: u32 = 0;
        let relocation_backup = (
            get_u32(&self.decompressed, reloc_dir),
            get_u32(&self.decompressed, reloc_dir.wrapping_add(4)),
        );
        'second_stage: for &key_offset in second_stage_key_offsets {
            let second_stage_key = get_u32(&self.decompressed, tbl.wrapping_add(key_offset));
            for zero_reloc in [false, true] {
                let relocation = if zero_reloc {
                    (0, 0)
                } else {
                    relocation_backup
                };
                write_u32(&mut self.decompressed, reloc_dir, relocation.0);
                write_u32(
                    &mut self.decompressed,
                    reloc_dir.wrapping_add(4),
                    relocation.1,
                );
                let mut hcs_addr = tbl.wrapping_add(if old_layout { 0x60 } else { 0x58 });
                header_checksum = 0;
                while get_u32(&self.decompressed, hcs_addr.wrapping_add(4)) != 0 {
                    header_checksum ^= self.calculate_checksum(hcs_addr);
                    hcs_addr = hcs_addr.wrapping_add(8);
                }
                let ss_key = header_checksum ^ first_stage_cs ^ second_stage_key;
                self.decompressed[ss_lo..ss_hi].copy_from_slice(&ss_ct);
                self.decrypt_data3(ss_pair, ss_key, 21);
                let pair = ss.wrapping_add(pair_off);
                let off = get_u32(&self.decompressed, pair) as u64;
                let size = get_u32(&self.decompressed, pair.wrapping_add(4)) as u64;
                if off > 0x1000 && off < len && size >= 4 && off.saturating_add(size) <= len {
                    found = true;
                    break 'second_stage;
                }
                if old_layout
                    && layout::find_pe32_dp_base(&self.decompressed, ss, ss_size).is_some()
                {
                    found = true;
                    break 'second_stage;
                }
            }
        }
        if !found {
            return Err(UnpackError::Pe32RelocationDataNotFound);
        }
        if verbose {
            println!(
                "  ss = 0x{:08X}, size = 0x{:X}, shift = 0x{:X}",
                ss, ss_size, ss_shift
            );
        }

        // Earlier layouts carry variable padding at the end of SecondStage, so
        // find the repeated stage descriptors structurally there.
        let dp_base_off = if old_layout {
            layout::find_pe32_dp_base(&self.decompressed, ss, ss_size)
                .ok_or(UnpackError::Pe32ThirdStageFailed)?
        } else {
            0xA9Cu32.wrapping_add(ss_shift)
        };
        let third_pair_off = if old_layout {
            dp_base_off.wrapping_add(0xF0)
        } else {
            0xB8Cu32.wrapping_add(ss_shift)
        };

        // ---- Stage 4: ThirdStage (trial key location and rotate shift) ----
        let pair_addr = ss.wrapping_add(third_pair_off);
        let ts_addr = get_u32(&self.decompressed, pair_addr);
        let ts_size_raw = get_u32(&self.decompressed, pair_addr.wrapping_add(4));
        if ts_addr <= 0x1000
            || (ts_addr as usize).saturating_add(ts_size_raw as usize) > self.decompressed.len()
            || ts_size_raw < 0x20
        {
            return Err(UnpackError::Pe32ThirdStageFailed);
        }
        let backup: Vec<u8> =
            self.decompressed[ts_addr as usize..(ts_addr + ts_size_raw) as usize].to_vec();
        let mut key_offsets = vec![0x968u32.wrapping_add(ss_shift)];
        if old_layout {
            let mut offset = 0u32;
            while offset + 4 <= ss_size {
                if offset != key_offsets[0]
                    && get_u32(&self.decompressed, ss.wrapping_add(offset)) != 0
                {
                    key_offsets.push(offset);
                }
                offset = offset.wrapping_add(4);
            }
        }
        let checksum_descriptors_valid = |data: &[u8], key_offset: u32| -> bool {
            let image_len = data.len() as u64;
            (0..3u32).all(|index| {
                let descriptor = ss
                    .wrapping_add(key_offset)
                    .wrapping_add(4 + index.wrapping_mul(8));
                if descriptor as usize + 8 > data.len() {
                    return false;
                }
                let offset = get_u32(data, descriptor) as u64;
                let size = get_u32(data, descriptor.wrapping_add(4)) as u64;
                offset > 0x1000 && size > 0 && offset.saturating_add(size) <= image_len
            })
        };
        let mut info_table: Option<u32> = None;
        let mut keys_addr: u32 = 0;
        let mut ts: u32 = 0;
        let mut third_key_off = key_offsets[0];
        'third_stage: for &key_offset in &key_offsets {
            let key = get_u32(&self.decompressed, ss.wrapping_add(key_offset));
            for &shift in &[19u32, 21, 17, 23, 15, 25, 13, 11] {
                self.decompressed[ts_addr as usize..(ts_addr + ts_size_raw) as usize]
                    .copy_from_slice(&backup);
                write_u32(&mut self.decompressed, pair_addr, ts_addr);
                write_u32(
                    &mut self.decompressed,
                    pair_addr.wrapping_add(4),
                    ts_size_raw,
                );
                self.decrypt_data3(pair_addr, key, shift);
                let mut off = 0u32;
                while off + 32 < ts_size_raw {
                    let t0 = get_u32(&self.decompressed, ts_addr.wrapping_add(off));
                    if t0 == 1 || t0 == 0x11 {
                        let t1 = get_u32(&self.decompressed, ts_addr.wrapping_add(off + 16));
                        if t1 == 2 {
                            let addr0 = get_u32(&self.decompressed, ts_addr.wrapping_add(off + 4));
                            if 0x1000 < addr0
                                && (addr0 as usize) < self.decompressed.len()
                                && checksum_descriptors_valid(&self.decompressed, key_offset)
                            {
                                let table = ts_addr.wrapping_add(off);
                                info_table = Some(table);
                                keys_addr = table.wrapping_sub(0x58);
                                ts = ts_addr;
                                third_key_off = key_offset;
                                break 'third_stage;
                            }
                        }
                    }
                    off = off.wrapping_add(4);
                }
            }
        }
        let info_table = info_table.ok_or(UnpackError::Pe32ThirdStageFailed)?;
        let cs_base_off = third_key_off.wrapping_add(4);
        let forth_key_off = third_key_off.wrapping_sub(4);
        if verbose {
            println!("[4/9] Decrypting stages (PE32)...");
            println!(
                "  thirdStage start = 0x{:X}, infoTable = 0x{:X}",
                ts, info_table
            );
            println!(
                "  third_key @ ss+0x{:X}, dp_base @ ss+0x{:X}",
                third_key_off, dp_base_off
            );
        }

        // ---- Process infoTable ----
        let mut it_addr = info_table;
        let mut metadata_pointer = 0u32;
        for _ in 0..2 {
            let tval = get_u32(&self.decompressed, it_addr);
            if tval == 1 || tval == 0x11 {
                metadata_pointer = get_u32(&self.decompressed, it_addr.wrapping_add(4));
                self.decrypt_data4(it_addr.wrapping_add(4));
            } else if tval == 2 {
                let mut copy_addr = get_u32(&self.decompressed, it_addr.wrapping_add(4));
                loop {
                    self.decrypt_data5(copy_addr, 16);
                    let s_a = get_u32(&self.decompressed, copy_addr);
                    let s_sz = get_u32(&self.decompressed, copy_addr.wrapping_add(4));
                    let d_a = get_u32(&self.decompressed, copy_addr.wrapping_add(8));
                    let d_sz = get_u32(&self.decompressed, copy_addr.wrapping_add(12));
                    copy_addr = copy_addr.wrapping_add(16);
                    if s_sz == 0 {
                        break;
                    }
                    if s_a != 0 && d_a != 0 && d_sz == s_sz {
                        let sa = s_a as usize;
                        let da = d_a as usize;
                        let n = s_sz as usize;
                        self.decompressed.copy_within(sa..sa + n, da);
                    }
                }
            }
            it_addr = it_addr.wrapping_add(16);
        }

        // ---- keyOffsets ----
        let mut ka = keys_addr;
        for k in 0..2usize {
            let mut ka2 = ka;
            for l in 0..2usize {
                self.decrypt_data4(ka2);
                self.key_offsets[k * 2 + l] = get_u32(&self.decompressed, ka2);
                ka2 = ka2.wrapping_add(8);
            }
            ka = ka.wrapping_add(32);
        }
        if verbose {
            println!(
                "  keys_addr = 0x{:X}, key_offsets = [0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}]",
                keys_addr,
                self.key_offsets[0],
                self.key_offsets[1],
                self.key_offsets[2],
                self.key_offsets[3]
            );
        }

        // ---- Checksum addresses ----
        let second_stage_cs_addr = tbl.wrapping_add(0xB0);
        let forth_stage_cs_addr = ss.wrapping_add(cs_base_off);
        let fifth_stage_cs_addr = ss.wrapping_add(cs_base_off).wrapping_add(0x08);
        let seven_stage_cs_addr = ss.wrapping_add(cs_base_off).wrapping_add(0x10);

        // ---- ForthStage ----
        let second_stage_cs = self.calculate_checksum(second_stage_cs_addr);
        let dp_base = ss.wrapping_add(dp_base_off);
        let forth_key_rounds = if old_layout {
            2
        } else {
            stage_key_rounds(&self.decompressed, dp_base, 4)?
        };
        let forth_addr = dp_base.wrapping_add(0x40);
        let forth_raw = get_u32(&self.decompressed, ss.wrapping_add(forth_key_off));
        let forth_src = get_u32(&self.decompressed, forth_addr) as usize;
        let forth_size = get_u32(&self.decompressed, forth_addr.wrapping_add(4)) as usize;
        let forth_descriptor = forth_addr as usize;
        if forth_src.saturating_add(forth_size) > self.decompressed.len()
            || forth_descriptor.saturating_add(16) > self.decompressed.len()
        {
            return Err(UnpackError::Pe32ThirdStageFailed);
        }
        let forth_backup = self.decompressed[forth_src..forth_src + forth_size].to_vec();
        let forth_descriptor_backup =
            self.decompressed[forth_descriptor..forth_descriptor + 16].to_vec();
        let mut forth_failure = None;
        let mut forth_ok = false;
        for rounds in [forth_key_rounds, 1, 2, 3, 4, 5, 6] {
            self.decompressed[forth_src..forth_src + forth_size].copy_from_slice(&forth_backup);
            self.decompressed[forth_descriptor..forth_descriptor + 16]
                .copy_from_slice(&forth_descriptor_backup);
            let key = header_checksum ^ second_stage_cs ^ advance_key(forth_raw, rounds);
            match self.decrypt_and_decompress_data(forth_addr, key, None) {
                Ok(()) => {
                    if verbose {
                        println!("  forth key walk rounds = {rounds}");
                    }
                    forth_ok = true;
                    break;
                }
                Err(reason) => forth_failure = Some(reason),
            }
        }
        if !forth_ok {
            if let Some(reason) = forth_failure {
                return Err(UnpackError::StageDecompressionFailed {
                    stage: DecompressionStage::Pe32FourthStage,
                    reason,
                });
            }
            return Err(UnpackError::Pe32ThirdStageFailed);
        }

        // ---- FifthStage ----
        let fifth_addr = dp_base.wrapping_add(0x50);
        let forth_cs = self.calculate_checksum(forth_stage_cs_addr);
        let forth_region_off = get_u32(&self.decompressed, forth_stage_cs_addr);
        let forth_region_sz = get_u32(&self.decompressed, forth_stage_cs_addr.wrapping_add(4));
        let fifth_key = get_u32(
            &self.decompressed,
            forth_region_off
                .wrapping_add(forth_region_sz)
                .wrapping_sub(4),
        );
        let fk5 = header_checksum ^ forth_cs ^ fifth_key;
        if let Err(reason) = self.decrypt_and_decompress_data(fifth_addr, fk5, None) {
            return Err(UnpackError::StageDecompressionFailed {
                stage: DecompressionStage::Pe32FifthStage,
                reason,
            });
        }

        // ---- SevenStage ----
        let seven_addr = dp_base.wrapping_add(0x70);
        let seven_dsz = get_u32(&self.decompressed, seven_addr.wrapping_add(12));
        let fifth_cs = self.calculate_checksum(fifth_stage_cs_addr);
        let cs1_addr = get_u32(
            &self.decompressed,
            ss.wrapping_add(cs_base_off).wrapping_add(0x08),
        );
        let cs1_size = get_u32(
            &self.decompressed,
            ss.wrapping_add(cs_base_off)
                .wrapping_add(0x08)
                .wrapping_add(4),
        );
        let seven_src = get_u32(&self.decompressed, seven_addr) as usize;
        let seven_size = get_u32(&self.decompressed, seven_addr.wrapping_add(4)) as usize;
        let seven_descriptor = seven_addr as usize;
        if seven_src.saturating_add(seven_size) > self.decompressed.len()
            || seven_descriptor.saturating_add(16) > self.decompressed.len()
        {
            return Err(UnpackError::Pe32ThirdStageFailed);
        }
        let seven_backup = self.decompressed[seven_src..seven_src + seven_size].to_vec();
        let seven_descriptor_backup =
            self.decompressed[seven_descriptor..seven_descriptor + 16].to_vec();
        let mut seven_failure = None;
        let mut seven_ok = false;
        'seventh_stage: for gap in [
            0x10u32, 0x04, 0x08, 0x0C, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28,
        ] {
            let key_offset = cs1_addr.wrapping_add(cs1_size).wrapping_sub(gap);
            if key_offset as usize + 4 > self.decompressed.len() {
                continue;
            }
            let raw_key = get_u32(&self.decompressed, key_offset);
            for invert in [true, false] {
                self.decompressed[seven_src..seven_src + seven_size].copy_from_slice(&seven_backup);
                self.decompressed[seven_descriptor..seven_descriptor + 16]
                    .copy_from_slice(&seven_descriptor_backup);
                let stage_key = if invert { !raw_key } else { raw_key };
                let key = header_checksum ^ fifth_cs ^ stage_key;
                match self.decrypt_and_decompress_data(seven_addr, key, None) {
                    Ok(()) => {
                        if verbose {
                            println!("  seven key gap = 0x{gap:X}, inverted = {invert}");
                        }
                        seven_ok = true;
                        break 'seventh_stage;
                    }
                    Err(reason) => seven_failure = Some(reason),
                }
            }
        }
        if !seven_ok {
            if let Some(reason) = seven_failure {
                return Err(UnpackError::StageDecompressionFailed {
                    stage: DecompressionStage::Pe32SeventhStage,
                    reason,
                });
            }
            return Err(UnpackError::Pe32ThirdStageFailed);
        }

        // ---- EighthStage ----
        let seven_start_actual = get_u32(&self.decompressed, seven_addr);
        if verbose {
            println!("[5/9] Decrypting eighthStage (PE32)...");
            println!(
                "  sevenStart = 0x{:X}, sevenDsz = 0x{:X}",
                seven_start_actual, seven_dsz
            );
        }
        // Locate the customDecryptor LFSR block (scan backward from middle, then
        // forward as fallback).
        let scan_start = seven_dsz / 2;
        let custom_dec_off = layout::find_lfsr_block(
            &self.decompressed,
            seven_start_actual,
            seven_dsz,
            scan_start,
            true,
        )
        .or_else(|| {
            layout::find_lfsr_block(&self.decompressed, seven_start_actual, seven_dsz, 0, false)
        })
        .ok_or(UnpackError::Pe32CustomDecryptorNotFound)?;
        let custom_dec_addr = seven_start_actual.wrapping_add(custom_dec_off);
        self.decrypt_data6(custom_dec_addr);
        let custom_ops = generate(&self.decompressed, custom_dec_addr).ok_or(
            UnpackError::BytecodeGenerationFailed(BytecodeStage::Pe32CustomDecryptor),
        )?;

        let seven_cs = self.calculate_checksum(seven_stage_cs_addr);
        let eighth_key_rounds = if old_layout {
            2
        } else {
            stage_key_rounds(&self.decompressed, dp_base.wrapping_add(0x80), 4)?
        };
        if verbose {
            println!("  eighth-stage key rounds = {eighth_key_rounds}");
        }
        let eighth_addr = dp_base.wrapping_add(0xC0);
        let eighth_dsz = get_u32(&self.decompressed, eighth_addr.wrapping_add(12));
        let eighth_src = get_u32(&self.decompressed, eighth_addr);
        let eighth_ssz = get_u32(&self.decompressed, eighth_addr.wrapping_add(4));
        let eighth_backup: Vec<u8> =
            self.decompressed[eighth_src as usize..(eighth_src + eighth_ssz) as usize].to_vec();
        let eighth_pair_bak: Vec<u8> =
            self.decompressed[eighth_addr as usize..(eighth_addr + 16) as usize].to_vec();
        let data_len = self.decompressed.len() as u32;

        // Build the eighthStageKey candidate list (offsets relative to
        // sevenStart) using gap heuristics + scan.
        let mut candidates: Vec<u32> = Vec::new();
        let push_cand = |c: &mut Vec<u32>, off: u32| {
            if !c.contains(&off) {
                c.push(off);
            }
        };
        for &end_gap in &[0xD0u32, 0xC0, 0xE0, 0xB0, 0xA0, 0xF0, 0x100] {
            if end_gap <= seven_dsz {
                let off = seven_dsz - end_gap;
                if off < seven_dsz {
                    let val = get_u32(&self.decompressed, seven_start_actual.wrapping_add(off));
                    if val != 0 && val != 0xCCCC_CCCC {
                        push_cand(&mut candidates, off);
                    }
                }
            }
        }
        for &gap in &[
            0x70u32, 0xD0, 0x28, 0x50, 0x48, 0x30, 0x40, 0x58, 0x60, 0x20, 0x38, 0x80, 0x90, 0xA0,
            0xB0,
        ] {
            if gap <= custom_dec_off {
                let off = custom_dec_off - gap;
                if off + 4 <= seven_dsz && !candidates.contains(&off) {
                    let val = get_u32(&self.decompressed, seven_start_actual.wrapping_add(off));
                    if val != 0 && val != 0xCCCC_CCCC {
                        push_cand(&mut candidates, off);
                    }
                }
            }
        }
        let scan_lo = custom_dec_off.saturating_sub(0x100);
        let mut off = scan_lo;
        while off < custom_dec_off {
            if !candidates.contains(&off) {
                let val = get_u32(&self.decompressed, seven_start_actual.wrapping_add(off));
                let all_printable = (0..4u32).all(|i| {
                    let b = (val >> (i * 8)) & 0xFF;
                    (32..127).contains(&b)
                });
                if val != 0 && val != 0xCCCC_CCCC && !all_printable {
                    push_cand(&mut candidates, off);
                }
            }
            off = off.wrapping_add(4);
        }

        let k1 = self.key_offsets[1];
        let k3 = self.key_offsets[3];
        let mut eighth_ok = false;
        for ek_off in candidates {
            self.decompressed[eighth_src as usize..(eighth_src + eighth_ssz) as usize]
                .copy_from_slice(&eighth_backup);
            self.decompressed[eighth_addr as usize..(eighth_addr + 16) as usize]
                .copy_from_slice(&eighth_pair_bak);
            let raw = get_u32(&self.decompressed, seven_start_actual.wrapping_add(ek_off));
            let test_key = advance_key(raw, eighth_key_rounds);
            let fk8 = header_checksum ^ fifth_cs ^ seven_cs ^ test_key;
            let result = primitives::decrypt_and_decompress_data(
                &mut self.decompressed,
                eighth_addr,
                fk8,
                k1,
                k3,
                Some(&custom_ops),
            );
            if result {
                let est = get_u32(&self.decompressed, eighth_addr);
                if 0x1000 < est && est < data_len {
                    eighth_ok = true;
                    break;
                }
            }
        }
        if !eighth_ok {
            return Err(UnpackError::Pe32EighthKeyNotFound);
        }
        let eighth_start = get_u32(&self.decompressed, eighth_addr);
        if verbose {
            println!(
                "  eighthStart = 0x{:08X}, dsz = 0x{:X}",
                eighth_start, eighth_dsz
            );
        }

        // ---- Final processing offsets (anchored on the eighthStage config cluster) ----
        //
        // The eighthStage holds a config cluster — importTable, fileCS,
        // compressedInfo, zeroList — at fixed offsets from a cluster base
        // (base+0x18 / +0x30 / +0x40 / +0x48) with the fileLFSR at +0x4B4.
        // Classic builds stamp a 0x00007679 dword at that base; native DLLs and
        // some older PE32 EXEs (ss_size=0xBE8) omit the stamp.
        // Locate the cluster by stamp when present (validated by fileCS at
        // base+0x30 pointing past info[3]); otherwise fall back to finding the
        // fileCS slot by shape — (addr, size) with addr just past info[3] and a
        // small 16-aligned size — and back-derive base = fileCS_off - 0x30.
        // Hardcoded eighthStart-relative constants remain as a last-resort
        // fallback for builds where neither discovery path fires.
        let marker = {
            let mut m: Option<u32> = None;
            let hi = eighth_dsz.saturating_sub(0x4C);
            let mut o = 0u32;
            while o < hi {
                if get_u32(&self.decompressed, eighth_start.wrapping_add(o)) == 0x7679 {
                    let fc = get_u32(&self.decompressed, eighth_start.wrapping_add(o + 0x30));
                    if fc > info3 && (fc as usize) < self.decompressed.len() {
                        m = Some(o);
                        break;
                    }
                }
                o = o.wrapping_add(4);
            }
            for anchor in [info3, info[6]] {
                if m.is_some() {
                    break;
                }
                let info3 = anchor;
                // fileCS-shaped slot: addr in (info3, info3+0x2000], size in
                // 0x10..=0x200 and 16-aligned. Prefer the candidate whose addr
                // is closest to (but past) info3 — matches every observed
                // build (one PE32 EXE family dist ~0x1C0, another ~0x1A0).
                let mut best: Option<(u32 /*dist*/, u32 /*off*/)> = None;
                let mut o = 0u32;
                let dlen = self.decompressed.len() as u32;
                while o + 8 <= eighth_dsz {
                    let fc = get_u32(&self.decompressed, eighth_start.wrapping_add(o));
                    let sz = get_u32(&self.decompressed, eighth_start.wrapping_add(o + 4));
                    if fc > info3
                        && fc <= info3.wrapping_add(0x2000)
                        && fc < dlen
                        && (0x10..=0x200).contains(&sz)
                        && (sz & 0xF) == 0
                    {
                        // Compact stages place the decryptor closer to this
                        // cluster. Only the config fields must fit here; the
                        // actual LFSR location is trial-validated below.
                        if o >= 0x30 {
                            let base = o - 0x30;
                            if base.wrapping_add(0x4C) <= eighth_dsz {
                                let dist = fc - info3;
                                match best {
                                    None => best = Some((dist, base)),
                                    Some((bd, _)) if dist < bd => best = Some((dist, base)),
                                    _ => {}
                                }
                            }
                        }
                    }
                    o = o.wrapping_add(4);
                }
                if let Some((dist, base)) = best {
                    if verbose {
                        println!(
                            "  pe32 cluster via fileCS (no 0x7679): base=+0x{:X} dist_info3=0x{:X}",
                            base, dist
                        );
                    }
                    m = Some(base);
                }
            }
            if m.is_none() {
                let image_len = self.decompressed.len() as u32;
                let mut offset = 0x18u32;
                while offset + 8 <= eighth_dsz {
                    let base = offset - 0x18;
                    if base + 0x38 > eighth_dsz {
                        break;
                    }
                    let rva = get_u32(&self.decompressed, eighth_start.wrapping_add(offset));
                    let size = get_u32(&self.decompressed, eighth_start.wrapping_add(offset + 4));
                    let imports_valid = rva > 0x1000
                        && rva < image_len
                        && size >= 40
                        && size.is_multiple_of(20)
                        && rva.wrapping_add(size) < image_len;
                    if imports_valid {
                        let checksum =
                            get_u32(&self.decompressed, eighth_start.wrapping_add(base + 0x30));
                        let checksum_size =
                            get_u32(&self.decompressed, eighth_start.wrapping_add(base + 0x34));
                        if checksum > 0x1000
                            && checksum < image_len
                            && (0x10..=0x200).contains(&checksum_size)
                            && checksum_size.is_multiple_of(16)
                        {
                            if verbose {
                                println!(
                                    "  pe32 cluster via import dir: base=+0x{base:X} imports=(0x{rva:X}, 0x{size:X})"
                                );
                            }
                            m = Some(base);
                            break;
                        }
                    }
                    offset = offset.wrapping_add(4);
                }
            }
            m
        };
        if verbose {
            match marker {
                Some(offset) => println!("  eighth cluster marker = +0x{offset:X}"),
                None => println!("  eighth cluster marker = none"),
            }
        }
        let (
            off_import_table,
            mut off_file_cs,
            mut off_compressed_info,
            mut off_zero_list,
            off_file_lfsr,
        ) = match marker {
            Some(m) => (m + 0x18, m + 0x30, m + 0x40, m + 0x48, m + 0x4B4),
            None => (
                0x3C50u32.wrapping_add(ss_shift),
                0x3C68u32.wrapping_add(ss_shift),
                0x3C78u32.wrapping_add(ss_shift),
                0x3C80u32.wrapping_add(ss_shift),
                0x40ECu32.wrapping_add(ss_shift),
            ),
        };
        let compressed_info_default = off_compressed_info;
        self.compressed_data_offset =
            (!get_u32(self.file_data, self.header_table_offset.wrapping_add(0x80)))
                .wrapping_add(self.header_table_offset);

        // ---- File checksums (permanent decrypt) ----
        let file_cs_addr_ptr = eighth_start.wrapping_add(off_file_cs);
        let mut file_cs_addr = get_u32(&self.decompressed, file_cs_addr_ptr);
        let file_cs_size = get_u32(&self.decompressed, file_cs_addr_ptr.wrapping_add(4));
        if file_cs_size > 0 {
            let file_cs_end = file_cs_addr.wrapping_add(file_cs_size);
            while file_cs_addr < file_cs_end {
                self.decrypt_data5(file_cs_addr, 16);
                file_cs_addr = file_cs_addr.wrapping_add(16);
            }
        } else {
            while get_u32(&self.decompressed, file_cs_addr.wrapping_add(4)) != 0 {
                self.decrypt_data5(file_cs_addr, 16);
                file_cs_addr = file_cs_addr.wrapping_add(16);
            }
        }

        // ---- File decryptor LFSR ----
        //
        // When the marker-relative off_file_lfsr is in range, try that slot
        // first (exact). If it is not a valid LFSR block, trial-and-validate
        // candidates from off_zero_list forward — required for older PE32 EXEs
        // without the 0x7679 stamp where the expected slot is empty and a loose
        // decoded[0]+0xC3 nearest-hit picks the wrong decryptor. Fall back to
        // the legacy loose scan only if no candidate trial-decompresses. Native
        // DLLs have a smaller eighthStage where off_file_lfsr lands out of range
        // and use the same trial-validate scan from just past the cluster (else
        // branch).
        let lfsr_off = if off_file_lfsr.wrapping_add(96) <= eighth_dsz {
            let mut lfsr_off = off_file_lfsr;
            let exact = layout::find_lfsr_block(
                &self.decompressed,
                eighth_start,
                eighth_dsz,
                off_file_lfsr,
                false,
            );
            if exact != Some(off_file_lfsr) {
                // Prefer trial-and-validate (same as the DLL branch): a loose
                // decoded[0]+0xC3 scan can land on coincidental LFSR-shaped
                // blocks that decode to a wrong file_ops and scramble every
                // compressed block. Observed on older PE32 EXEs without the
                // 0x7679 cluster stamp: the expected slot is empty and the
                // nearest loose hit is not the real decryptor.
                let ci_slot = eighth_start.wrapping_add(off_compressed_info);
                let mut chosen: Option<u32> = None;
                let mut considered = 0u32;
                'file_decryptor: for start in [off_zero_list, 0] {
                    let mut scan = start;
                    while let Some(candidate) = layout::find_lfsr_block(
                        &self.decompressed,
                        eighth_start,
                        eighth_dsz,
                        scan,
                        false,
                    ) {
                        considered = considered.wrapping_add(1);
                        if self
                            .pe32_file_lfsr_validates(eighth_start.wrapping_add(candidate), ci_slot)
                        {
                            chosen = Some(candidate);
                            break 'file_decryptor;
                        }
                        scan = candidate.wrapping_add(1);
                    }
                }
                if let Some(c) = chosen {
                    if verbose {
                        println!(
                            "  pe32 fileLFSR via trial-validate: +0x{:X} (expected +0x{:X}, considered {})",
                            c, off_file_lfsr, considered
                        );
                    }
                    lfsr_off = c;
                } else {
                    // No candidate trial-decompresses: fail loudly. The old
                    // "legacy loose scan" picked the nearest LFSR-shaped block
                    // by offset distance without any validation — that is
                    // exactly how a wrong file_ops got applied to every data
                    // block (uncompressed blocks never enter the decompressor),
                    // producing a plausible but fully wrong image (the PE32
                    // .text scramble root cause). Trial-and-validate or error.
                    return Err(UnpackError::Pe32FileLfsrNotFound);
                }
            }
            lfsr_off
        } else {
            // Native DLL: the marker-relative off_file_lfsr (EXE-tuned, marker +
            // 0x4B4) overshoots the smaller DLL eighthStage, so the exact slot is
            // unavailable. A plain forward scan returns the FIRST valid-opcode
            // block, but the DLL eighthStage contains coincidental valid-opcode
            // blocks that decode to trivial programs (e.g. a constant byte add)
            // ahead of the real file decryptor. A wrong file_ops corrupts the
            // per-block translate (applied before decompression), so every data
            // block fails to decompress. Enumerate every candidate forward and
            // keep the first whose decoded file_ops actually decompresses the
            // first compressed data block — trial-and-validate, same idea as the
            // D1/D2 fixes. Non-DLL (EXE) builds never reach this branch.
            let mut chosen: Option<u32> = None;
            let classic_offset =
                (!get_u32(self.file_data, self.header_table_offset.wrapping_add(0x80)))
                    .wrapping_add(self.header_table_offset);
            'file_decryptor: for info_offset in
                [off_compressed_info, off_compressed_info.wrapping_sub(8)]
            {
                let info_slot = eighth_start.wrapping_add(info_offset);
                let table = get_u32(&self.decompressed, info_slot);
                let mut data_offsets = vec![classic_offset];
                if table as usize + 4 <= self.decompressed.len() {
                    let first_source = trial_decrypt5_u32(&self.decompressed, table);
                    if let Some(file_offset) = prot_rva_to_off(self.file_data, pe_off, first_source)
                    {
                        let derived = file_offset.wrapping_sub(first_source);
                        if derived != classic_offset {
                            data_offsets.push(derived);
                        }
                    }
                }
                for data_offset in data_offsets {
                    self.compressed_data_offset = data_offset;
                    for start in [off_zero_list.wrapping_add(8), 0] {
                        let mut scan = start;
                        while let Some(candidate) = layout::find_lfsr_block(
                            &self.decompressed,
                            eighth_start,
                            eighth_dsz,
                            scan,
                            false,
                        ) {
                            if self.pe32_file_lfsr_validates(
                                eighth_start.wrapping_add(candidate),
                                info_slot,
                            ) {
                                off_compressed_info = info_offset;
                                chosen = Some(candidate);
                                break 'file_decryptor;
                            }
                            scan = candidate.wrapping_add(1);
                        }
                    }
                }
            }
            if verbose && let Some(offset) = chosen {
                println!(
                    "  fileLFSR +0x{offset:X}, compressedInfo slot +0x{off_compressed_info:X}"
                );
            }
            chosen.ok_or(UnpackError::Pe32FileLfsrNotFound)?
        };
        if off_compressed_info != compressed_info_default {
            let delta = compressed_info_default.wrapping_sub(off_compressed_info);
            off_file_cs = off_file_cs.wrapping_sub(delta);
            off_zero_list = off_zero_list.wrapping_sub(delta);
            if verbose {
                println!(
                    "  cluster tail shifted by -0x{delta:X}: fileCS +0x{off_file_cs:X}, zeroList +0x{off_zero_list:X}"
                );
            }
        }
        let file_dec_addr = eighth_start.wrapping_add(lfsr_off);
        self.decrypt_data6(file_dec_addr);
        let file_ops = generate(&self.decompressed, file_dec_addr).ok_or(
            UnpackError::BytecodeGenerationFailed(BytecodeStage::Pe32FileDecryptor),
        )?;

        // ---- PE32 metadata: EP and data dirs from info[3] ----
        let plain_shell = self.header_table_offset != super::super::super::HEADER_TABLE_OFF;
        let info3 = if plain_shell && metadata_pointer > 0x1000 {
            metadata_pointer.wrapping_sub(0x10)
        } else {
            info3
        };
        let test_val = get_u32(&self.decompressed, info3.wrapping_add(0x10));
        let metadata_ep: u32;
        let mut metadata_dirs = [0u8; 128];
        let metadata_shift = if plain_shell {
            0
        } else if old_layout {
            0x10
        } else {
            0
        };
        if test_val > 0x10000 {
            // Layout B
            let s = info3.wrapping_add(0x10) as usize;
            let backup_meta = self.decompressed[s..s + 0x290].to_vec();
            self.decrypt_data5(info3.wrapping_add(0x10), 0x290);
            metadata_ep = get_u32(
                &self.decompressed,
                info3.wrapping_add(0x20 + metadata_shift),
            );
            let d = info3.wrapping_add(0x30 + metadata_shift) as usize;
            metadata_dirs.copy_from_slice(&self.decompressed[d..d + 128]);
            self.decompressed[s..s + 0x290].copy_from_slice(&backup_meta);
        } else {
            // Layout A
            let s = info3.wrapping_add(0x40) as usize;
            let backup_meta = self.decompressed[s..s + 144].to_vec();
            self.decrypt_data5(info3.wrapping_add(0x40), 144);
            metadata_ep = get_u32(
                &self.decompressed,
                info3.wrapping_add(0x40 + metadata_shift),
            );
            let d = info3.wrapping_add(0x50 + metadata_shift) as usize;
            metadata_dirs.copy_from_slice(&self.decompressed[d..d + 128]);
            self.decompressed[s..s + 144].copy_from_slice(&backup_meta);
        }

        // ---- Zero-out list (runs BEFORE decompression) ----
        let zero_list_addr = eighth_start.wrapping_add(off_zero_list);
        let mut zero_ptr = get_u32(&self.decompressed, zero_list_addr);
        loop {
            self.decrypt_data5(zero_ptr, 16);
            let src3 = get_u32(&self.decompressed, zero_ptr);
            let s_sz3 = get_u32(&self.decompressed, zero_ptr.wrapping_add(4));
            zero_ptr = zero_ptr.wrapping_add(16);
            if s_sz3 == 0 {
                break;
            }
            if src3.wrapping_add(s_sz3) as usize > self.decompressed.len() {
                break;
            }
            for b in &mut self.decompressed[src3 as usize..(src3 + s_sz3) as usize] {
                *b = 0;
            }
        }

        // ---- File data decompression ----
        if verbose {
            println!("[6/9] Loading and decompressing file data (PE32)...");
        }
        let mut compress_data_offset = self.compressed_data_offset;
        let compressed_info_addr = eighth_start.wrapping_add(off_compressed_info);
        let mut compressed_info = get_u32(&self.decompressed, compressed_info_addr);
        // Pass 1 (sequential): position-keyed descriptor chain (decrypt_data5),
        // terminated by a zero source-size record.
        struct Blk {
            src: u32,
            ssz: u32,
            dst: u32,
            dsz: u32,
        }
        let mut blocks: Vec<Blk> = Vec::new();
        let compressed_info_bytes =
            get_u32(&self.decompressed, compressed_info_addr.wrapping_add(4));
        let max_blocks = if compressed_info_bytes >= 16 {
            (compressed_info_bytes / 16) as usize
        } else {
            1 << 20
        };
        loop {
            if blocks.len() >= max_blocks
                || (compressed_info as usize).saturating_add(16) > self.decompressed.len()
            {
                break;
            }
            self.decrypt_data5(compressed_info, 16);
            let src2 = get_u32(&self.decompressed, compressed_info);
            let s_sz2 = get_u32(&self.decompressed, compressed_info.wrapping_add(4));
            let dst2 = get_u32(&self.decompressed, compressed_info.wrapping_add(8));
            let d_sz2 = get_u32(&self.decompressed, compressed_info.wrapping_add(12));
            compressed_info = compressed_info.wrapping_add(16);
            if s_sz2 == 0 {
                break;
            }
            blocks.push(Blk {
                src: src2,
                ssz: s_sz2,
                dst: dst2,
                dsz: d_sz2,
            });
        }
        if plain_shell {
            let resolved: Option<Vec<u32>> = blocks
                .iter()
                .map(|block| prot_rva_to_off(self.file_data, pe_off, block.src))
                .collect();
            if let Some(file_offsets) = resolved {
                for (block, file_offset) in blocks.iter_mut().zip(file_offsets) {
                    block.src = file_offset;
                }
                compress_data_offset = 0;
            }
        }
        if verbose {
            println!("  blocks = {}", blocks.len());
        }
        let stub_restore = if plain_shell {
            find_stub_restore(self.file_data, pe_off)
        } else {
            None
        };
        if verbose && let Some(restore) = stub_restore {
            println!(
                "  stub restore: {:#x}..{:#x} <- {:#x}",
                restore.dst,
                restore.dst.wrapping_add(restore.len),
                restore.src
            );
        }
        // Pass 2: independent per-block work over disjoint dst spans (see
        // `parallel_for` for how the spans are carved safely).
        {
            let lut = OpsLut::new(&file_ops);
            let clean = &self.file_data;
            let ko = self.key_offsets;
            let ks_snap = primitives::aes_schedule_snapshot(&self.decompressed, ko[2])
                .ok_or(UnpackError::InvalidAesKeySchedule { offset: ko[2] })?;
            let tab_snap = primitives::huffman_table_snapshot(&self.decompressed, ko[0])
                .ok_or(UnpackError::InvalidHuffmanTable { offset: ko[0] })?;
            let spans: Vec<(usize, usize)> = blocks
                .iter()
                .map(|b| {
                    let s = b.dst as usize;
                    (s, s + b.ssz.max(b.dsz) as usize)
                })
                .collect();
            let do_block = |i: usize, base: usize, span: &mut [u8]| -> Result<(), UnpackError> {
                let b = &blocks[i];
                let file_src = b.src.wrapping_add(compress_data_offset) as usize;
                let rel = b.dst as usize - base;
                let n = b.ssz as usize;
                span[rel..rel + n].copy_from_slice(&clean[file_src..file_src + n]);
                if let Some(restore) = stub_restore {
                    let overlap_start = file_src.max(restore.dst as usize);
                    let overlap_end =
                        (file_src + n).min(restore.dst as usize + restore.len as usize);
                    if overlap_start < overlap_end {
                        let destination = rel + (overlap_start - file_src);
                        let source = restore.src as usize + (overlap_start - restore.dst as usize);
                        let len = overlap_end - overlap_start;
                        span[destination..destination + len]
                            .copy_from_slice(&clean[source..source + len]);
                    }
                }
                primitives::aes_decrypt_ks(&ks_snap, span, rel as u32, b.ssz);
                lut.map_region(span, rel, n);
                if b.ssz != b.dsz {
                    // decompress reports corruption (after partial writes) via
                    // its bool; surface it instead of shipping a garbage block.
                    if !primitives::decompress_tbl(
                        &tab_snap, span, rel as u32, rel as u32, b.ssz, b.dsz,
                    ) {
                        return Err(UnpackError::SectionDecompressionFailed {
                            pipeline: SectionPipeline::ExePe32,
                            block: i,
                        });
                    }
                }
                Ok(())
            };
            super::super::super::parallel::parallel_for(
                &mut self.decompressed,
                &spans,
                1,
                do_block,
            )?;
        }

        // ---- Section fixup ----
        self.decompressed[..0x1000].copy_from_slice(&self.file_data[..0x1000]);
        let opt_hdr_size = get_u16(self.file_data, pe_off.wrapping_add(20)) as u32;
        let sec_hdr_table = pe_off.wrapping_add(24).wrapping_add(opt_hdr_size);
        let export_va = get_u32(
            self.file_data,
            pe_off
                .wrapping_add(24)
                .wrapping_add(opt_hdr_size)
                .wrapping_sub(128),
        );
        let export_size = get_u32(
            self.file_data,
            pe_off
                .wrapping_add(24)
                .wrapping_add(opt_hdr_size)
                .wrapping_sub(124),
        );
        let mut export_file_off: u32 = 0;
        let mut text_off: u32 = 0;
        let mut text_size: u32 = 0;
        // Walk by NumberOfSections (PE has no zero-VS sentinel; a real
        // VirtualSize==0 section would truncate these fixups early), stopping
        // at the all-zero padding in case NumberOfSections is overstated.
        let num_sections = get_u16(self.file_data, pe_off.wrapping_add(6)) as u32;
        for i in 0..num_sections.min(96) {
            let sec_hdr = sec_hdr_table.wrapping_add(i.wrapping_mul(40));
            if self.file_data[sec_hdr as usize..sec_hdr as usize + 8]
                .iter()
                .all(|&b| b == 0)
            {
                break;
            }
            let va = get_u32(self.file_data, sec_hdr.wrapping_add(12));
            let sz = get_u32(self.file_data, sec_hdr.wrapping_add(8));
            let f_off = get_u32(self.file_data, sec_hdr.wrapping_add(20));
            let name = section_name(self.file_data, sec_hdr);
            if name.starts_with(".text") {
                text_size = sz;
                text_off = va;
            }
            if export_size != 0
                && export_va >= va
                && export_va.wrapping_add(export_size) <= va.wrapping_add(sz)
            {
                export_file_off = export_va.wrapping_sub(va).wrapping_add(f_off);
            }
            write_u32(&mut self.decompressed, sec_hdr.wrapping_add(16), sz);
            write_u32(&mut self.decompressed, sec_hdr.wrapping_add(20), va);
            if name.starts_with(".idata") {
                write_u32(
                    &mut self.decompressed,
                    sec_hdr.wrapping_add(36),
                    0xC000_0040,
                );
            }
        }
        if export_size != 0 && export_file_off != 0 {
            let d = export_va as usize;
            let s = export_file_off as usize;
            let n = export_size as usize;
            // Spliced companions may omit the stub's plaintext exports;
            // the I/O adapter overlays those from the original stub later.
            if s.checked_add(n)
                .is_some_and(|end| end <= self.file_data.len())
                && d.checked_add(n)
                    .is_some_and(|end| end <= self.decompressed.len())
            {
                self.decompressed[d..d + n].copy_from_slice(&self.file_data[s..s + n]);
            }
        }

        // ---- .text decrypt with decrypt_data8 (PE32 auto-detected formula) ----
        // `select_dd8_formula_pe32` returns None when `.text` was not packer-dd8-
        // encrypted (native DLLs leave it plaintext); applying dd8 there would
        // scramble valid code, so skip it entirely in that case.
        if text_size > 0 && text_off > 0 {
            if let Some(big) =
                layout::select_dd8_formula_pe32(&self.decompressed, text_off, text_size)
            {
                if verbose {
                    println!(
                        "[7/9] Decrypting .text (PE32 dd8, formula={})...",
                        if big { "0x8000*(page+1)" } else { "page+1" }
                    );
                }
                let num_pages = text_size / 0x1000;
                for page in 0..num_pages {
                    let pk = if big {
                        0x8000u32.wrapping_mul(page.wrapping_add(1))
                    } else {
                        page.wrapping_add(1)
                    };
                    let pa = text_off.wrapping_add(page.wrapping_mul(0x1000));
                    let mut k = pk;
                    let rk = k.rotate_right(15);
                    k = rk;
                    for bi in 1..256u32 {
                        let rk = k.rotate_right(15);
                        let ri = rk.wrapping_add(bi);
                        k = ri.wrapping_add(bi);
                        let tidx =
                            pa.wrapping_add(bi.wrapping_mul(16)).wrapping_add(ri & 0xF) as usize;
                        self.decompressed[tidx] ^= k as u8;
                    }
                }
            } else if verbose {
                println!("[7/9] Skipping .text dd8 (already plaintext)...");
            }
        }

        // ---- Fix data directories (PE32: data dirs at pe+0x78) ----
        let exe_pe = get_u32(&self.decompressed, 60);
        for i in 0..128u32 {
            self.decompressed[(exe_pe + 0x78 + i) as usize] = metadata_dirs[i as usize];
        }
        // DLL-aware reloc / DllCharacteristics handling. An EXE's packer rebuilds
        // the relocation table and clears DllCharacteristics, so the loader needs
        // no relocations. A DLL, by contrast, is almost always mapped at a
        // non-preferred base, so it MUST keep its base-relocation directory
        // (restored above from metadata_dirs) and a valid DllCharacteristics
        // (DYNAMIC_BASE) — zeroing them leaves the DLL unrelocatable and its
        // imports pinned to the packer stub, so it fails to load (which looks
        // like a missing/broken export table).
        let is_dll = (get_u16(&self.decompressed, exe_pe.wrapping_add(22)) & 0x2000) != 0;
        if !is_dll {
            // EXE: clear BaseReloc (index 5 = pe+0xA0) and DllCharacteristics (pe+0x5E).
            write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xA0), 0);
            write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xA4), 0);
            write_u16(&mut self.decompressed, exe_pe.wrapping_add(0x5E), 0);
        } else {
            // DLL: keep the BaseReloc dir from metadata; ensure DYNAMIC_BASE.
            let mut dll_chars = get_u16(&self.decompressed, exe_pe.wrapping_add(0x5E));
            if dll_chars == 0 {
                dll_chars = 0x0040; // IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            }
            write_u16(
                &mut self.decompressed,
                exe_pe.wrapping_add(0x5E),
                dll_chars as u32,
            );
        }

        // ---- TLS directory reconstruction (PE32: index 9 = pe+0xC0) ----
        let tls_dir_rva = get_u32(&self.decompressed, exe_pe.wrapping_add(0xC0));
        let tls_dir_sz = get_u32(&self.decompressed, exe_pe.wrapping_add(0xC4));
        if tls_dir_rva > 0
            && tls_dir_sz >= 24
            && (tls_dir_rva as usize + 24) <= self.decompressed.len()
        {
            let all_zero = (0..6u32)
                .all(|i| get_u32(&self.decompressed, tls_dir_rva.wrapping_add(i * 4)) == 0);
            if all_zero {
                let image_base = get_u32(&self.decompressed, exe_pe.wrapping_add(52));
                // Prefer the module's real TLS directory, which survives in the
                // loader stub's plaintext `.rdata`/`.tls`. Only when the stub
                // cannot supply one does a placeholder get synthesized: it keeps
                // the image loadable, but drops the initialized TLS template,
                // `_tls_index` and the TLS callback array, so any module that
                // actually uses `thread_local` faults once it runs.
                if !self.restore_pe32_tls_from_stub(pe_off, tls_dir_rva, image_base) {
                    let mut tls_sec_va: u32 = 0;
                    let mut data_sec_va: u32 = 0;
                    let mut data_sec_sz: u32 = 0;
                    let sh = exe_pe
                        .wrapping_add(24)
                        .wrapping_add(get_u16(&self.decompressed, exe_pe.wrapping_add(20)) as u32);
                    let ns = get_u16(&self.decompressed, exe_pe.wrapping_add(6)) as u32;
                    for i in 0..ns {
                        let s = sh.wrapping_add(i * 40);
                        let nm = get_string_to_null(&self.decompressed, s);
                        let va = get_u32(&self.decompressed, s.wrapping_add(12));
                        let sz = get_u32(&self.decompressed, s.wrapping_add(16));
                        if nm.starts_with(".tls") {
                            tls_sec_va = va;
                        }
                        if nm.starts_with(".data") {
                            data_sec_va = va;
                            data_sec_sz = sz;
                        }
                    }
                    if tls_sec_va > 0 && data_sec_va > 0 {
                        let start_raw = image_base.wrapping_add(tls_sec_va);
                        let end_raw = start_raw;
                        let idx_addr = image_base
                            .wrapping_add(data_sec_va)
                            .wrapping_add(data_sec_sz)
                            .wrapping_sub(16);
                        let cb_addr = image_base
                            .wrapping_add(data_sec_va)
                            .wrapping_add(data_sec_sz)
                            .wrapping_sub(8);
                        let scratch = (data_sec_va + data_sec_sz - 16) as usize;
                        for b in &mut self.decompressed[scratch..scratch + 16] {
                            *b = 0;
                        }
                        write_u32(&mut self.decompressed, tls_dir_rva, start_raw);
                        write_u32(&mut self.decompressed, tls_dir_rva.wrapping_add(4), end_raw);
                        write_u32(
                            &mut self.decompressed,
                            tls_dir_rva.wrapping_add(8),
                            idx_addr,
                        );
                        write_u32(
                            &mut self.decompressed,
                            tls_dir_rva.wrapping_add(12),
                            cb_addr,
                        );
                        write_u32(&mut self.decompressed, tls_dir_rva.wrapping_add(16), 0);
                        write_u32(
                            &mut self.decompressed,
                            tls_dir_rva.wrapping_add(20),
                            0x30_0000,
                        );
                    } else {
                        write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xC0), 0);
                        write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xC4), 0);
                    }
                }
            }
        }

        // ---- Re-arm TLS field base relocations (PE32 DLL) ----
        // The packer neutralizes the four relocations covering the TLS
        // directory's VA fields (Start/EndAddressOfRawData, AddressOfIndex,
        // AddressOfCallBacks) by demoting them to IMAGE_REL_BASED_ABSOLUTE
        // padding, because its own loader fixes TLS up by hand. Restored
        // verbatim, a DLL mapped off its preferred base keeps stale VAs in its
        // TLS directory, and the OS loader faults in LdrpAllocateTlsEntry when
        // it writes the TLS slot index through the unrelocated AddressOfIndex.
        // Promote those entries back to IMAGE_REL_BASED_HIGHLOW. EXE output
        // is untouched: its BaseReloc directory is zeroed above, and its
        // goldens must stay byte-identical.
        if is_dll && tls_dir_rva > 0 {
            let reloc_rva = get_u32(&self.decompressed, exe_pe.wrapping_add(0xA0));
            let reloc_size = get_u32(&self.decompressed, exe_pe.wrapping_add(0xA4));
            let reloc_end = reloc_rva.wrapping_add(reloc_size);
            let dlen = self.decompressed.len() as u32;
            if reloc_rva > 0 && reloc_size >= 8 && reloc_end <= dlen {
                let mut rearmed = 0u32;
                let mut block = reloc_rva;
                while block.wrapping_add(8) <= reloc_end {
                    let page = get_u32(&self.decompressed, block);
                    let block_size = get_u32(&self.decompressed, block.wrapping_add(4));
                    if block_size < 8
                        || !block_size.is_multiple_of(2)
                        || block.wrapping_add(block_size) > reloc_end
                    {
                        break;
                    }
                    let entries = (block_size - 8) / 2;
                    for i in 0..entries {
                        let entry_off = block.wrapping_add(8).wrapping_add(i.wrapping_mul(2));
                        let entry = get_u16(&self.decompressed, entry_off);
                        let entry_rva = page.wrapping_add((entry & 0x0FFF) as u32);
                        // IMAGE_REL_BASED_ABSOLUTE with a real offset is
                        // neutralized, not padding (padding keeps offset 0).
                        if entry >> 12 == 0
                            && entry & 0x0FFF != 0
                            && entry_rva >= tls_dir_rva
                            && entry_rva < tls_dir_rva.wrapping_add(16)
                        {
                            write_u16(&mut self.decompressed, entry_off, (entry | 0x3000) as u32);
                            rearmed = rearmed.wrapping_add(1);
                        }
                    }
                    block = block.wrapping_add(block_size);
                }
                if verbose && rearmed > 0 {
                    println!("  re-armed {} TLS field relocations", rearmed);
                }
            }
        }

        // ---- Import table (PE32, 4-byte thunks) ----
        if verbose {
            println!("[8/9] Decrypting import strings (PE32)...");
        }
        let import_table_addr = eighth_start.wrapping_add(off_import_table);
        let mut import_table_ptr = get_u32(&self.decompressed, import_table_addr);
        let mut idt_size = get_u32(&self.decompressed, import_table_addr.wrapping_add(4));

        let metadata_import_rva = get_u32(&metadata_dirs, 8);
        let metadata_import_size = get_u32(&metadata_dirs, 12);
        let dlen = self.decompressed.len() as u32;

        let mut eighth_import_valid = false;
        if 0 < import_table_ptr && import_table_ptr < dlen && 0 < idt_size && idt_size < 0x10000 {
            let test_name = if import_table_ptr + 20 <= dlen {
                get_u32(&self.decompressed, import_table_ptr.wrapping_add(12))
            } else {
                0
            };
            let test_ilt = if import_table_ptr + 4 <= dlen {
                get_u32(&self.decompressed, import_table_ptr)
            } else {
                0
            };
            if 0x1000 < test_name && test_name < dlen && 0x1000 < test_ilt && test_ilt < dlen {
                eighth_import_valid = true;
            }
        }
        let mut metadata_import_valid = false;
        if 0x1000 < metadata_import_rva && metadata_import_rva < dlen.wrapping_sub(20) {
            let test_name2 = get_u32(&self.decompressed, metadata_import_rva.wrapping_add(12));
            let test_ilt2 = get_u32(&self.decompressed, metadata_import_rva);
            if 0x1000 < test_name2 && test_name2 < dlen && 0x1000 < test_ilt2 && test_ilt2 < dlen {
                metadata_import_valid = true;
            }
        }
        if metadata_import_valid
            && (!eighth_import_valid || metadata_import_rva != import_table_ptr)
        {
            import_table_ptr = metadata_import_rva;
            idt_size = metadata_import_size;
        }

        if 0 < import_table_ptr && import_table_ptr < dlen && 0 < idt_size && idt_size < 0x10000 {
            let mut idt_pos = import_table_ptr;
            let idt_end = import_table_ptr.wrapping_add(idt_size);
            while idt_pos.wrapping_add(20) <= idt_end {
                let ilt_rva = get_u32(&self.decompressed, idt_pos);
                let name_rva = get_u32(&self.decompressed, idt_pos.wrapping_add(12));
                let iat_rva = get_u32(&self.decompressed, idt_pos.wrapping_add(16));
                if ilt_rva == 0 && name_rva == 0 && iat_rva == 0 {
                    break;
                }
                if 0 < name_rva && name_rva < dlen {
                    self.decrypt_data7(name_rva, name_rva as u8);
                }
                let thunk_base = if 0 < ilt_rva && ilt_rva < dlen {
                    ilt_rva
                } else {
                    iat_rva
                };
                if 0 < thunk_base && thunk_base < dlen.wrapping_sub(4) {
                    let mut thunk_pos = thunk_base;
                    while thunk_pos.wrapping_add(4) <= dlen {
                        let thunk_val = get_u32(&self.decompressed, thunk_pos);
                        if thunk_val == 0 {
                            break;
                        }
                        if thunk_val & 0x8000_0000 == 0 && thunk_val.wrapping_add(2) < dlen {
                            self.decrypt_data7(thunk_val.wrapping_add(2), thunk_val as u8);
                            write_u16(&mut self.decompressed, thunk_val, 0);
                        }
                        thunk_pos = thunk_pos.wrapping_add(4);
                    }
                }
                idt_pos = idt_pos.wrapping_add(20);
            }
        }

        // Update PE header: Import directory (index 1 = pe+0x80), clear IAT
        // directory (index 12 = pe+0xD8).
        write_u32(
            &mut self.decompressed,
            exe_pe.wrapping_add(0x80),
            import_table_ptr,
        );
        write_u32(&mut self.decompressed, exe_pe.wrapping_add(0x84), idt_size);
        write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xD8), 0);
        write_u32(&mut self.decompressed, exe_pe.wrapping_add(0xDC), 0);

        // ---- EP (from metadata) ----
        if metadata_ep > 0 {
            write_u32(&mut self.decompressed, exe_pe.wrapping_add(40), metadata_ep);
        } else {
            let real_ep = get_u32(self.file_data, pe_off.wrapping_add(40));
            write_u32(&mut self.decompressed, exe_pe.wrapping_add(40), real_ep);
        }

        // ---- Output transforms ----
        if verbose {
            println!("[9/9] Rebuilding PE file layout (PE32)...");
        }
        let mut out = std::mem::take(&mut self.decompressed);
        // kmiat import relocation is an EXE-only fixup: it discards the original
        // import directory in favour of the loader-written IAT stub. A DLL keeps
        // its real import table (restored above from metadata), so skip kmiat for
        // DLLs (`!is_dll` guard).
        let is_dll = (get_u16(&out, pe_off.wrapping_add(22)) & 0x2000) != 0;
        if !is_dll && !layout::pe32_imports_already_match_idata_layout(&mut out, pe_off) {
            layout::move_pe32_imports_to_kmiat(&mut out, pe_off);
        }
        let compact = layout::compact_memory_image_to_pe(&out, pe_off)
            .ok_or(UnpackError::Pe32OutputLayoutInvalid)?;
        Ok(compact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_key_rounds_follow_active_descriptor_prefix() {
        for (sizes, expected) in [
            ([0, 0x45, 0x45, 0x45], 0),
            ([0x45, 4, 0x45, 0x45], 1),
            ([0x45, 0x45, 0x45, 3], 3),
            ([0x45, 0x45, 0x45, 0x45], 4),
        ] {
            let mut data = [0u8; 80];
            for (index, size) in sizes.into_iter().enumerate() {
                write_u32(&mut data, 16 + index as u32 * 16 + 4, size);
            }
            assert_eq!(stage_key_rounds(&data, 16, 4).unwrap(), expected);
        }
    }

    #[test]
    fn stage_key_rounds_reject_truncated_tables() {
        assert!(matches!(
            stage_key_rounds(&[0u8; 63], 0, 4),
            Err(UnpackError::BufferRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            stage_key_rounds(&[0u8; 64], u32::MAX, 4),
            Err(UnpackError::BufferRangeOutOfBounds { .. })
        ));
    }
}
