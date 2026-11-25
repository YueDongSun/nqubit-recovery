module Core_Ticker(clk, trg, set, tic);

`include "Functions.sv"

parameter Interval = 1;

parameter CntWid = GetSize(Interval);
parameter [CntWid : 0] Start = Ticker_Conv_Start(Interval, CntWid);
parameter [CntWid : 0] Carry = Ticker_Conv_Carry(Interval, CntWid);

input   clk;
input   trg;
input   set;
output  tic;

reg tic = Start[CntWid];
reg [CntWid-1 : 0] c0 = Start[CntWid-1:0];
reg [CntWid : 0] c1 = Carry;

always @ (posedge clk)
if (tic | set)
begin
	{tic, c0} <= Start;
	c1 <= Carry;
end
else if (trg)
begin
	{tic, c0} <= {tic, c0} ^ c1;
	c1 <= {c0 & c1[CntWid-1 : 0], 1'b1};
end

endmodule
