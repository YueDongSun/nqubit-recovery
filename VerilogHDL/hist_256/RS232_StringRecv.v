module RS232_StringRecv(clk, in, str, recv);

parameter Len = 1;
parameter BaudCnt = 159;
parameter StopBit = 1;

input   clk;
input   in;
output  [Len*8-1 : 0] str;
output  recv;

wire [7 : 0] byt;
wire tick;
wire brk;
RS232_ByteRecv #(BaudCnt, StopBit) BRCV(clk, in, byt, tick, brk);

reg [Len*8-1 : 0] str = 0;
always @ (posedge clk) if (tick) str <= {str, byt};

reg [Len-1 : 0] cnt = 2;
always @ (posedge clk) if (brk) cnt <= 2; else if (tick) cnt <= {cnt, cnt[Len-1]};

reg recv = 0;
always @ (posedge clk) recv <= tick & cnt[0];

endmodule
