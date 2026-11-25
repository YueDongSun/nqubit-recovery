# FPGA histogram firmware

Verilog for the FPGA-side coincidence-histogramming controller of the
W-state experiment (Altera Cyclone III `EP3C25Q240C8`, Quartus projects
included).

| Directory | Design |
|-----------|--------|
| `3qubit/` | 75-cell layout (3Q, 15x15 operation grid, 300 ns/cell) |
| `4qubit/` | 256-cell layout (4Q, 16x16 grid, 400 ns/cell) |
| `hist/` | Standalone histogram build |
| `hist_256/` | 256-cell histogram build |

Each directory is a self-contained Quartus project: RTL (`main.v`,
`AWGControl.v`, `Core_Ticker.v`, `HISTGRAM.v`, `InputChannel.v`, UART RX/TX
and RS-232 framing) plus the `.qsf/.qpf/.sdc` project and pin constraints.
The counter address maps (signal/idler/coincidence histogram banks,
multi-click and timing fields) are documented in
`../system_control/data_ui/logic.txt`.

License note: the original Quartus MegaWizard-generated wrappers
(`fifo*.v`, `pll*.v` beyond the shipped stubs, and the assignment-defaults
`.qdf`) were Altera-licensed tool outputs and are not redistributed here.
`fifo.v` ships as a hand-written inferred-RAM replacement (8192x32,
read-latency-one) and `pll.v` as a behavioral 50->100 MHz model whose
header records the exact ALTPLL parameters to regenerate the production
wrapper with the Quartus II 13.0 SP1 megafunction wizard (inclk0 50 MHz,
c0 multiply 2 / divide 1, duty 50%, NO_COMPENSATION, Cyclone III).

The host side talks over UART: `awg_reproduction` in
`../system_control/experiment_control/` sequences the FPGA, and the
`data_ui` serial UIs read the histogram frames back.
