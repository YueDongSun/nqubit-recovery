function integer GetSize;

input [63 : 0] Num;

integer S;

begin
	S = 0;
	while (Num > 0)
	begin
		S = S + 1;
		Num = Num >> 1;
	end
	if (S == 0) GetSize = 64; else GetSize = S;
end

endfunction

function [63 : 0] Ticker_Conv_Start;

input [63 : 0] Num;
input integer Wid;

integer i;
reg [63 : 0] t0;
reg [63 : 0] t1;

begin
	t0 = 0;
	t1 = 0;
	Num = (1 << Wid) + Wid - Num;
	for (i = 0; i <= Wid; i = i + 1)
	begin
		t0[i] = Num[i];
		t1[i] = (t0 < i) | (t0 >= (1 << i) + i);
	end
	Ticker_Conv_Start = t1;
end

endfunction

function [63 : 0] Ticker_Conv_Carry;

input [63 : 0] Num;
input integer Wid;

integer i;
reg [63 : 0] t0;
reg [63 : 0] t1;

begin
	t0 = 0;
	t1 = 0;
	Num = (1 << Wid) + Wid - Num;
	for (i = 0; i < Wid; i = i + 1)
	begin
		t0[i] = Num[i];
		t1[i+1] = (t0 == i);
	end
	t1[0] = 1'b1;
	Ticker_Conv_Carry = t1;
end

endfunction
