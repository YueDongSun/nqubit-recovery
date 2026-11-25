module InputChannel(clk, in, out);

input   clk;
input   in;
output  out;

reg t = 0;
reg out = 0;

always @ (posedge clk)
begin
	t <= in;
	out <= in & ~t;
end

endmodule
