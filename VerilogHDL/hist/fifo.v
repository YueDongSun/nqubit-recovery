// Synchronous FIFO, 8192 x 32 bit, read-latency-one (showahead OFF).
// Hand-written inferred-RAM replacement for the original Quartus
// MegaWizard scfifo wrapper (parameters preserved: lpm_numwords=8192,
// lpm_width=32, lpm_widthu=13, overflow/underflow checking ON,
// no aclr/sclr, no full/usedw outputs used by this design).
module fifo (
    clock,
    data,
    rdreq,
    wrreq,
    empty,
    q);

    input         clock;
    input  [31:0] data;
    input         rdreq;
    input         wrreq;
    output        empty;
    output [31:0] q;

    localparam DEPTH = 8192;
    localparam AW    = 13;

    reg [31:0] mem [0:DEPTH-1];
    reg [AW:0] wr_ptr, rd_ptr;

    wire wr_ok = wrreq && (wr_ptr[AW] == rd_ptr[AW] || wr_ptr[AW-1:0] != rd_ptr[AW-1:0]);
    wire rd_ok = rdreq && (wr_ptr != rd_ptr);

    always @(posedge clock) begin
        if (wr_ok) begin
            mem[wr_ptr[AW-1:0]] <= data;
            wr_ptr <= wr_ptr + 1'b1;
        end
        if (rd_ok)
            rd_ptr <= rd_ptr + 1'b1;
    end

    // Read latency of one cycle (matches lpm_showahead = "OFF"): the data
    // word is captured one clock after rdreq, aligned with snd <= rreq in
    // TIC.v's two-beat handshake.
    reg [31:0] q_reg;
    always @(posedge clock)
        if (rd_ok)
            q_reg <= mem[rd_ptr[AW-1:0]];

    assign q     = q_reg;
    assign empty = (wr_ptr == rd_ptr);

endmodule
