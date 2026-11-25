module RS232_StringSend(clk, start, str, out, sent);

parameter Len = 1;
parameter BaudCnt = 159;
parameter StopBit = 2;

input   clk;
input   start;
input   [Len*8-1 : 0] str;
output  out;
output  sent;

reg dstr = 0;
always @ (posedge clk) dstr <= start;

wire tick;
reg fbs = 0;
reg [Len-1 : 0] cnt = 2;
always @ (posedge clk) fbs <= (dstr | tick) & ~cnt[0];

reg dbs = 0;
always @ (posedge clk) dbs <= fbs;

reg [Len*8-1 : 0] bfr = 0;
always @ (posedge clk)
if (start) bfr <= str;
else if (dbs) bfr <= {bfr, 8'hFF};

wire [7 : 0] byt = bfr[Len*8-1:Len*8-8];

RS232_ByteSend #(BaudCnt, StopBit) BSND(clk, fbs, byt, out, tick);


always @ (posedge clk) if (tick) cnt <= {cnt, cnt[Len-1]};

wire sent = tick & cnt[0];

endmodule
