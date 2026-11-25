import numpy as np
import os

# ================= Parameters =================
wavelength = int(500e3)     # Waveform length (number of points)
sampling = 1e9              # 1 point per 1ns

# ================= Frequencies =================
f = np.linspace(85, 122.5, 16)          # 85:2.5:120, total 16 points
fx = np.tile(f, (16, 1))              # 16x16, same per row
fy = fx.T                             # Transpose, fy[i,j] = f[i]

# ================= Amplitudes =================
ASx = np.array([0.8, 0.8, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85])
ASy = np.array([0.8, 0.95, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 , 0.95])

AmpSx = np.tile(ASx, (16, 1))                    # Same per row
AmpSy = np.tile(ASy[:, np.newaxis], (1, 16))     # Same per column

# Note: Missing comma in original AIy, fixed to 16 values
AIx = np.array([0.85, 0.85, 0.8, 0.8, 0.85, 0.85, 0.85, 0.8, 0.8, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85])
AIy = np.array([0.7, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.85, 0.9, 0.9, 0.9, 0.85])

AmpIx = np.tile(AIx, (16, 1))
AmpIy = np.tile(AIy[:, np.newaxis], (1, 16))

AmpWx = np.ones((16, 16))
AmpWy = np.ones((16, 16))
AmpRx = np.ones((16, 16))
AmpRy = np.ones((16, 16))

# ================= Start points (calibrated delays) =================
WriteSeqStart = 10000
StartW  = 9420
StartWx = 3970
StartWy = 4800
StartSx = 1750
StartSy = 0

ReadSeqStart = 10000
StartR  = 10100
StartRx = 1950
StartRy = 0
StartIx = 800
StartIy = 5100

# ================= Data generation functions =================

def create_zero_waveforms():
    waveforms = {}
    for ch in ['W_CH1', 'W_CH2', 'W_CH3', 'W_CH4', 'R_CH1', 'R_CH2', 'R_CH3', 'R_CH4']:
        waveforms[f'{ch}_mark1'] = np.zeros(wavelength)
        waveforms[f'{ch}_mark2'] = np.zeros(wavelength)
        waveforms[f'{ch}_ana'] = np.zeros(wavelength)
    return waveforms


def generate_phase_cali_for_indy(Indy, Indxref=7):
    seq = create_zero_waveforms()
    write_start = WriteSeqStart
    read_start = ReadSeqStart

    # Write markers
    start_mw = write_start + StartW          # 1‑based start point
    stop_mw = start_mw + 1000                # 1‑based end point (inclusive)
    s_w = start_mw - 1                       # Convert to 0‑based
    e_w = stop_mw                            # Left closed, right open
    seq['W_CH1_mark2'][s_w:e_w] = 1
    seq['W_CH2_mark2'][s_w:e_w] = 1
    seq['W_CH3_mark2'][s_w:e_w] = 1
    seq['W_CH4_mark2'][s_w:e_w] = 1

    # Read markers
    start_mr = read_start + StartR
    stop_mr = start_mr + 1000
    s_r = start_mr - 1
    e_r = stop_mr
    seq['R_CH1_mark2'][s_r:e_r] = 1

    Dt = 1000
    Nphase = 18
    Index = 0

    for Indx in range(1, 17):               # 1:16
        for IndPhase in range(Nphase + 1):  # 0:18
            Index += 1
            phase = IndPhase / Nphase * 2 * np.pi

            # --- Write Pulse ---
            start_wp = write_start + StartW + (Index - 1) * Dt
            stop_wp = start_wp + 100                    # Length 101 points
            s_wp = start_wp - 1
            e_wp = stop_wp
            seq['W_CH1_mark1'][s_wp:e_wp] = 1

            # --- Write AOD X (map to W_CH1) ---
            start_wx = write_start + StartWx + (Index - 1) * Dt
            stop_wx = start_wx + 1000                   # Length 1001 points
            s_wx = start_wx - 1
            e_wx = stop_wx
            t_wx = np.arange(s_wx, e_wx) / sampling
            freq_ix = fx[Indy-1, Indx-1]                 # 1‑based -> 0‑based
            freq_ir = fx[Indy-1, Indxref-1]
            seq['W_CH1_ana'][s_wx:e_wx] = (
                AmpWx[Indy-1, Indx-1]     * np.sin(2 * np.pi * freq_ix * 1e6 * t_wx + phase) +
                AmpWx[Indy-1, Indxref-1]  * np.sin(2 * np.pi * freq_ir * 1e6 * t_wx)
            )

            # --- Write AOD Y (map to W_CH2) ---
            start_wy = write_start + StartWy + (Index - 1) * Dt
            stop_wy = start_wy + 1000
            s_wy = start_wy - 1
            e_wy = stop_wy
            t_wy = np.arange(s_wy, e_wy) / sampling
            freq_ir_y = fy[Indy-1, Indxref-1]            # Note Y direction uses fy
            seq['W_CH2_ana'][s_wy:e_wy] = (
                AmpWy[Indy-1, Indxref-1] * np.sin(2 * np.pi * freq_ir_y * 1e6 * t_wy)
            )

            # --- Signal AOD X (map to W_CH3) ---
            start_sx = write_start + StartSx + (Index - 1) * Dt
            stop_sx = start_sx + 1000
            s_sx = start_sx - 1
            e_sx = stop_sx
            t_sx = np.arange(s_sx, e_sx) / sampling
            seq['W_CH3_ana'][s_sx:e_sx] = (
                AmpSx[Indy-1, Indx-1]     * np.sin(2 * np.pi * freq_ix * 1e6 * t_sx) +
                AmpSx[Indy-1, Indxref-1]  * np.sin(2 * np.pi * freq_ir * 1e6 * t_sx)
            )   # No phase

            # --- Signal AOD Y (map to W_CH4) ---
            start_sy = write_start + StartSy + (Index - 1) * Dt
            stop_sy = start_sy + 1000
            s_sy = start_sy - 1
            e_sy = stop_sy
            t_sy = np.arange(s_sy, e_sy) / sampling
            seq['W_CH4_ana'][s_sy:e_sy] = (
                AmpSy[Indy-1, Indxref-1] * np.sin(2 * np.pi * freq_ir_y * 1e6 * t_sy)
            )

            # --- Read Pulse ---
            start_rp = read_start + StartR + (Index - 1) * Dt
            stop_rp = start_rp + 220                    # Length 221 points
            s_rp = start_rp - 1
            e_rp = stop_rp
            seq['R_CH2_mark1'][s_rp:e_rp] = 1

            # --- Read AOD X ---
            start_rx = read_start + StartRx + (Index - 1) * Dt
            stop_rx = start_rx + 1000
            s_rx = start_rx - 1
            e_rx = stop_rx
            t_rx = np.arange(s_rx, e_rx) / sampling
            seq['R_CH1_ana'][s_rx:e_rx] = (
                AmpRx[Indy-1, Indx-1]     * np.sin(2 * np.pi * freq_ix * 1e6 * t_rx + phase) +
                AmpRx[Indy-1, Indxref-1]  * np.sin(2 * np.pi * freq_ir * 1e6 * t_rx)
            )

            # --- Read AOD Y ---
            start_ry = read_start + StartRy + (Index - 1) * Dt
            stop_ry = start_ry + 1000
            s_ry = start_ry - 1
            e_ry = stop_ry
            t_ry = np.arange(s_ry, e_ry) / sampling
            seq['R_CH2_ana'][s_ry:e_ry] = (
                AmpRy[Indy-1, Indxref-1] * np.sin(2 * np.pi * freq_ir_y * 1e6 * t_ry)
            )

            # --- Idler AOD X ---
            start_ix = read_start + StartIx + (Index - 1) * Dt
            stop_ix = start_ix + 1000
            s_ix = start_ix - 1
            e_ix = stop_ix
            t_ix = np.arange(s_ix, e_ix) / sampling
            seq['R_CH3_ana'][s_ix:e_ix] = (
                AmpIx[Indy-1, Indx-1]     * np.sin(2 * np.pi * freq_ix * 1e6 * t_ix) +
                AmpIx[Indy-1, Indxref-1]  * np.sin(2 * np.pi * freq_ir * 1e6 * t_ix)
            )   # No phase

            # --- Idler AOD Y ---
            start_iy = read_start + StartIy + (Index - 1) * Dt
            stop_iy = start_iy + 1000
            s_iy = start_iy - 1
            e_iy = stop_iy
            t_iy = np.arange(s_iy, e_iy) / sampling
            seq['R_CH4_ana'][s_iy:e_iy] = (
                AmpIy[Indy-1, Indxref-1] * np.sin(2 * np.pi * freq_ir_y * 1e6 * t_iy)
            )

        # After each outer loop, increase starting offset
        read_start += 10000
        write_start += 10000

    return seq

# ================= Output TXT files =================
out_dir = os.path.join('AWG_Sequence', 'phase_Cali_sequence_files')
os.makedirs(out_dir, exist_ok=True)

def save_awg_3row(filename, ana, mark1, mark2, scale_ana=1.0):
    """Save three rows of data as tab-separated text file, format: '%.7f\t%d\t%d\r\n'"""
    data = np.column_stack((ana * scale_ana, mark1, mark2))
    np.savetxt(filename, data, fmt=['%.14f', '%d', '%d'], delimiter='\t', newline='\n')

Indxref = 7

for Indy in range(1, 17):
    seq = generate_phase_cali_for_indy(Indy, Indxref=Indxref)

    # Write channels (W)
    save_awg_3row(os.path.join(out_dir, f'AWG_W_CH1_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['W_CH1_ana'], seq['W_CH1_mark1'], seq['W_CH1_mark2'], scale_ana=0.5)
    save_awg_3row(os.path.join(out_dir, f'AWG_W_CH2_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['W_CH2_ana'], seq['W_CH2_mark1'], seq['W_CH2_mark2'])
    save_awg_3row(os.path.join(out_dir, f'AWG_W_CH3_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['W_CH3_ana'], seq['W_CH3_mark1'], seq['W_CH3_mark2'], scale_ana=0.5)
    save_awg_3row(os.path.join(out_dir, f'AWG_W_CH4_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['W_CH4_ana'], seq['W_CH4_mark1'], seq['W_CH4_mark2'])

    # Read channels (R)
    save_awg_3row(os.path.join(out_dir, f'AWG_R_CH1_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['R_CH1_ana'], seq['R_CH1_mark1'], seq['R_CH1_mark2'], scale_ana=0.5)
    save_awg_3row(os.path.join(out_dir, f'AWG_R_CH2_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['R_CH2_ana'], seq['R_CH2_mark1'], seq['R_CH2_mark2'])
    save_awg_3row(os.path.join(out_dir, f'AWG_R_CH3_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['R_CH3_ana'], seq['R_CH3_mark1'], seq['R_CH3_mark2'], scale_ana=0.5)
    save_awg_3row(os.path.join(out_dir, f'AWG_R_CH4_PHASEx_Indy{Indy}_Indxref{Indxref}.txt'),
                  seq['R_CH4_ana'], seq['R_CH4_mark1'], seq['R_CH4_mark2'])

print("All waveform files generated successfully.")