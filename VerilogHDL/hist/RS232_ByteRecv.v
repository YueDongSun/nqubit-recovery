module RS232_ByteRecv(clk, in, byt, recv, brk);

parameter Baud = 17;
parameter StopBit = 1;

parameter BaudCnt = Baud - 1;

input   clk;
input   in;
output  [7 : 0] byt;
output  recv;
output  brk;

reg idle = 1;
always @ (posedge clk) idle <= (idle & in) | recv;

wire tick;
Core_Ticker #(BaudCnt) BAUD(clk, 1'b1, idle, tick);

wire brk;
Core_Ticker #(BaudCnt<<20) RBRK(clk, idle, recv, brk);

wire smp;
Core_Ticker #((BaudCnt>>1)+2) SAMP(clk, in, tick, smp);

reg cbt = 0;
always @ (posedge clk) cbt <= (smp | cbt) & ~tick;

reg [7+StopBit : 0] tmp = 0;
wire [7 : 0] byt = tmp[7:0];
always @ (posedge clk) if (tick) tmp <= {cbt, tmp[7+StopBit:1]};

reg [8+StopBit : 0] cnt = 2;
always @ (posedge clk) if (tick) cnt <= {cnt, cnt[8+StopBit]};

reg recv = 0;
always @ (posedge clk) recv <= tick & cnt[0];

endmodule
