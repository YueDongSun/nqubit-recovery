module TIC(clk, in, out);

input wire clk;
input wire [1:0]in ;

output reg out;

reg [30:0]timer = 0;
reg [1:0]nin = 0;//negedge in;
always @(negedge clk)
begin
timer <= timer + 1;
nin <= in;
end

wire [31:0] timedata;
assign timedata = {nin[1], timer};

wire wreq;
assign wreq = nin[1] | nin[0];

wire empty;
wire [31:0]dataout;
reg rreq = 0;
fifo storedata(clk, timedata, rreq, wreq, empty, dataout);

reg buzy = 0;
wire sent;
reg snd = 0;
always @(negedge clk)
begin
if(sent) buzy <= 0;

if(~buzy & ~empty) 
begin
rreq <= 1;
buzy <= 1;
end
else rreq <= 0;

snd <= rreq;
end

RS232_StringSend #(4, 50) timesend(clk, snd, dataout, out, sent);


endmodule
