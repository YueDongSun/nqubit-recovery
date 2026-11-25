# The board oscillator drives sys_clk at 50 MHz.  The ALTPLL in pll.v creates
# the internal 100 MHz clock used by the experiment-control logic.
create_clock -name sys_clk -period 20.000 [get_ports {sys_clk}]
derive_pll_clocks
derive_clock_uncertainty
