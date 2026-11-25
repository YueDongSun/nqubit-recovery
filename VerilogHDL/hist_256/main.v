module LatticeWriteRead (
    sys_clk,
    triggerall,
    in,
    Write,
    Read,
    //Write2,
    //Read2,
    PumpingR,
    PumpingW,
    uart_rx,
    uart_tx,
    rst_n
);

input sys_clk;
input triggerall;
input rst_n;
input [4:0] in; // in[0]=signal, in[1]=idler

output Write, Read, PumpingR, PumpingW; //Write2, //Read2, // delete cntout

// ##############################################################

input uart_rx;
output uart_tx;

parameter CLK_FRE = 50; // MHz
localparam UART_CLK_FRE = CLK_FRE * 2; // c0 is generated as 100MHz
//parameter num_cellline=256;
// 1338-byte byte-aligned frame for 256 measurement bins plus 2-click pair counters

localparam IDLE = 0;
localparam SEND = 1; // send HELLO ALINX\r\n
localparam WAIT = 2; // wait 1 second and send uart received data

reg  [7:0]   tx_data;
reg           tx_data_valid;
wire          tx_data_ready;
reg  [15:0]   tx_cnt;
wire [7:0]    rx_data;
wire          rx_data_valid;
wire          rx_data_ready;
reg  [31:0]   wait_cnt;
reg  [3:0]    state;
reg           send_ctrl = 1'b0;
reg           tx_busy = 1'b0;

// ##############################################################

wire c0; // clock 10ns
//wire [30:0] timeall;
wire Write, Read;
wire hist_busy_counts;
wire hist_busy;
wire tx_request;
wire tx_next;
wire [7:0] counter_tx_byte;
//wire [2:0] triger;

assign rx_data_ready = 1'b1; // always can receive data

pll RunClk (
    .inclk0(sys_clk),
    .c0(c0) // double clock frequency 100MHz
);

assign tx_request = (state == IDLE) && send_ctrl && !hist_busy_counts;
assign hist_busy = hist_busy_counts | tx_busy | tx_request;
assign tx_next = tx_data_valid && tx_data_ready && (tx_cnt < 16'd1337);

wire [4:0] in1;
InputChannel IC1 (c0, in[0], in1[0]); // signal
InputChannel IC2 (c0, in[1], in1[1]); // idler
InputChannel IC3 (c0, in[2], in1[2]);
InputChannel IC4 (c0, in[3], in1[3]);
InputChannel IC5 (c0, in[4], in1[4]);

AWGControl control(
    .clk(c0),
    .triggerall(triggerall),
    .hist_busy(hist_busy),
    .Write(Write),
    .Read(Read),
    .PumpingW(PumpingW),
    .PumpingR(PumpingR)
    //.freeze_counts(freeze_counts),
    //.hist_updating(hist_updating),
    //.timeall(timeall)
);

HISTGRAM RunCnt (
    .clk(c0),
    .in(in1),
    .Write(Write),
    .Read(Read),
    .tx_start(tx_request),
    .tx_next(tx_next),
    .tx_byte(counter_tx_byte),
    .hist_busy(hist_busy_counts)
	 //.triggerall(triggerall),
    //.timeall(timeall)
    //.triger(triger)
);

// ##############################################################
always @(posedge c0 or negedge rst_n) begin
    if (rst_n == 1'b0)
        send_ctrl <= 0;
    else if (rx_data_valid && (rx_data == "y"))
        send_ctrl <= 1'b1;
    else if (rx_data_valid && (rx_data == "n"))
        send_ctrl <= 0;
    else
        send_ctrl <= send_ctrl;
end

always @(posedge c0 or negedge rst_n) begin
    if (rst_n == 1'b0) begin
        state         <= IDLE;
        tx_busy       <= 1'b0;
        tx_cnt        <= 16'd0;
        tx_data       <= 8'd0;
        tx_data_valid <= 1'b0;
        wait_cnt      <= 32'd0;
    end
    else case (state)
        IDLE: begin
            wait_cnt      <= 32'd0;
            tx_cnt        <= 16'd0;
            tx_data       <= 8'd0;
            tx_data_valid <= 1'b0;
            tx_busy       <= 1'b0;

            if (send_ctrl && !hist_busy_counts) begin
                state  <= SEND;
                tx_busy <= 1'b1;
            end
            else begin
                state <= state;
            end
        end

        SEND: begin
            tx_data <= counter_tx_byte;

            if (tx_next) begin
                tx_cnt <= tx_cnt + 16'd1; // send data counter
            end
            else if (tx_data_valid && tx_data_ready) begin
                // last byte sent is complete
                tx_cnt        <= 16'd0;
                tx_data_valid <= 1'b0;
                tx_busy       <= 1'b0;
                state         <= WAIT;
            end
            else if (~tx_data_valid) begin
                tx_data_valid <= 1'b1;
            end
        end

        WAIT: begin
            wait_cnt <= wait_cnt + 32'd1;
            if (wait_cnt >= UART_CLK_FRE * 1000000) // wait for 1 second
                state <= IDLE;
            else
                state <= state;
        end

        default: state <= IDLE;
    endcase
end

/***************************************************************************
 * Calling uart_tx module and uart_rx module
 ***************************************************************************/

uart_rx #(
    .CLK_FRE(UART_CLK_FRE),
    .BAUD_RATE(500000)
) uart_rx_inst (
    .clk(c0),
    .rst_n(rst_n),
    .rx_data(rx_data),
    .rx_data_valid(rx_data_valid),
    .rx_data_ready(rx_data_ready),
    .rx_pin(uart_rx)
);

uart_tx #(
    .CLK_FRE(UART_CLK_FRE),
    .BAUD_RATE(500000)
) uart_tx_inst (
    .clk(c0),
    .rst_n(rst_n),
    .tx_data(tx_data),
    .tx_data_valid(tx_data_valid),
    .tx_data_ready(tx_data_ready),
    .tx_pin(uart_tx)
);

//Optical_pumping//


//assign Write = triggerall;
//assign Read = triggerall;

endmodule
