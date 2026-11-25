module RS232_ByteSend(clk, start, byt, out, sent);

parameter Baud = 17;
parameter StopBit = 1;
 
parameter BaudCnt = Baud - 1;

input   clk;
input   start;
input   [7 : 0] byt;
output  out;
output  sent;

wire tick;
Core_Ticker #(BaudCnt) BAUD(clk, 1'b1, start, tick);

reg [8 : 0] bfr = 9'h1FF;
wire out = bfr[0];
always @ (posedge clk) 
if (start) bfr <= {byt, 1'b0}; 
else if (tick) bfr <= {1'b1, bfr[8:1]};

reg [8+StopBit : 0] cnt = 0;
always @ (posedge clk)
if (start) cnt <= 1;
else if (tick) cnt <= cnt << 1;

wire sent = tick & cnt[8+StopBit];

endmodule
