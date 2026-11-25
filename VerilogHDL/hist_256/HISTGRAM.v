module HISTGRAM(clk, in, Write, Read, tx_start, tx_next, tx_byte, hist_busy);

    input clk;
    input [4:0] in;
    input Write, Read;
    input tx_start;
    input tx_next;

    output [7:0] tx_byte;
    output hist_busy;

    parameter signals = 0;
    parameter signale = 40;
    parameter idlers = 7;
    parameter idlere = 27;
    parameter ws_aoddelay = 1000;  // max write signal aod delay
    parameter ri_aoddelay = 1100; // max read idler aod delay
    parameter sdetectwindow = (signale - signals) / 10;
    parameter idetectwindow = (idlere - idlers) / 10;
    parameter cellhist = 12000;

    localparam integer CELL_COUNT = 256;
    localparam integer CELL_PERIOD = 100;
    localparam integer HIST_BINS = 10;
    localparam integer FRAME_BYTES = 1338;
    localparam integer WRITE_SEQ_TICK = CELL_COUNT * CELL_PERIOD + ws_aoddelay;
    localparam integer READ_AGG_TICK = CELL_COUNT * CELL_PERIOD + ri_aoddelay + 100;

    localparam TX_IDLE = 4'd0;
    localparam TX_READ_COUNT = 4'd1;
    localparam TX_WRITE_COUNT = 4'd2;
    localparam TX_IDLER_ALL = 4'd3;
    localparam TX_COIN_ALL = 4'd4;
    localparam TX_SIGNAL_ALL = 4'd5;
    localparam TX_SIGNAL_CELL = 4'd6;
    localparam TX_IDLER_CELL = 4'd7;
    localparam TX_COIN_CELL = 4'd8;
    localparam TX_SIGNAL_HIST = 4'd9;
    localparam TX_IDLER_HIST = 4'd10;
    localparam TX_MULTI2_PAIR = 4'd11;

    reg hist_busy = 1'b0;
    reg [7:0] tx_byte = 8'd0;
    reg [3:0] tx_state = TX_IDLE;
    reg [1:0] tx_sub = 2'd0;
    reg [7:0] tx_cell = 8'd0;
    reg [3:0] tx_hist = 4'd0;
    reg [2:0] tx_pair = 3'd0;

    reg [18:0] read_count = 19'd0;
    reg [18:0] write_count = 19'd0;
    reg [18:0] signal_all = 19'd0;
    reg [18:0] idler_all = 19'd0;
    reg [14:0] coin_all = 15'd0;
    reg [31:0] multi2_12_count = 32'd0;
    reg [31:0] multi2_13_count = 32'd0;
    reg [31:0] multi2_14_count = 32'd0;
    reg [31:0] multi2_23_count = 32'd0;
    reg [31:0] multi2_24_count = 32'd0;
    reg [31:0] multi2_34_count = 32'd0;

    reg [10:0] signal_cell [0:CELL_COUNT-1];
    reg [10:0] idler_cell [0:CELL_COUNT-1];
    reg [6:0] coin_cell [0:CELL_COUNT-1];
    reg [7:0] signal_hist [0:HIST_BINS-1];
    reg [7:0] idler_hist [0:HIST_BINS-1];

    reg [15:0] time1 = 16'd0;
    reg [15:0] time2 = 16'd0;

    reg Write_d = 1'b0;

    reg write_scan = 1'b0;
    reg read_scan = 1'b0;
    reg [7:0] write_cell = 8'd0;
    reg [7:0] read_cell = 8'd0;
    reg [6:0] write_phase = 7'd0;
    reg [6:0] read_phase = 7'd0;

    reg [255:0] getsignal = 256'd0;
    reg [255:0] getidler = 256'd0;

    reg agg_active = 1'b0;
    reg agg_flush = 1'b0;
    reg agg_write_valid = 1'b0;
    reg agg_done = 1'b0;
    reg [7:0] agg_cell = 8'd0;
    reg [7:0] agg_write_cell = 8'd0;
    reg agg_signal_hit = 1'b0;
    reg agg_idler_hit = 1'b0;
    reg [10:0] agg_signal_old = 11'd0;
    reg [10:0] agg_idler_old = 11'd0;
    reg [6:0] agg_coin_old = 7'd0;

    integer hist_i;
    integer init_i;

    wire write_start = Write & ~Write_d;
    wire signal_hit = (Write == 1'b1 && in[0] == 1'b1 && write_scan &&
                       write_phase > signals && write_phase < signale);
    wire [2:0] idler_clicks;
    wire idler_hit = (Read == 1'b1 && idler_clicks != 3'd0 && read_scan &&
                      read_phase > idlers && read_phase < idlere);
    assign idler_clicks = {2'b0, in[1]} + {2'b0, in[2]} +
                          {2'b0, in[3]} + {2'b0, in[4]};

    initial begin
        for (init_i = 0; init_i < CELL_COUNT; init_i = init_i + 1) begin
            signal_cell[init_i] = 11'd0;
            idler_cell[init_i] = 11'd0;
            coin_cell[init_i] = 7'd0;
        end
        for (init_i = 0; init_i < HIST_BINS; init_i = init_i + 1) begin
            signal_hist[init_i] = 8'd0;
            idler_hist[init_i] = 8'd0;
        end
    end

    function [7:0] counter19_byte;
        input [18:0] value;
        input [1:0] byte_sel;
        begin
            case (byte_sel)
                2'd0: counter19_byte = {5'd0, value[18:16]};
                2'd1: counter19_byte = value[15:8];
                default: counter19_byte = value[7:0];
            endcase
        end
    endfunction

    function [7:0] counter15_byte;
        input [14:0] value;
        input byte_sel;
        begin
            if (byte_sel == 1'b0)
                counter15_byte = {1'b0, value[14:8]};
            else
                counter15_byte = value[7:0];
        end
    endfunction

    function [7:0] counter32_byte;
        input [31:0] value;
        input [1:0] byte_sel;
        begin
            case (byte_sel)
                2'd0: counter32_byte = value[31:24];
                2'd1: counter32_byte = value[23:16];
                2'd2: counter32_byte = value[15:8];
                default: counter32_byte = value[7:0];
            endcase
        end
    endfunction

    always @(posedge clk) begin
        Write_d <= Write;

        if (tx_start) begin
            tx_state <= TX_READ_COUNT;
            tx_sub <= 2'd1;
            tx_cell <= 8'd0;
            tx_hist <= 4'd0;
            tx_pair <= 3'd0;
            tx_byte <= counter19_byte(read_count, 2'd0);
        end else if (tx_next) begin
            case (tx_state)
                TX_READ_COUNT: begin
                    tx_byte <= counter19_byte(read_count, tx_sub);
                    if (tx_sub == 2'd2) begin
                        tx_state <= TX_WRITE_COUNT;
                        tx_sub <= 2'd0;
                    end else begin
                        tx_sub <= tx_sub + 2'd1;
                    end
                end

                TX_WRITE_COUNT: begin
                    tx_byte <= counter19_byte(write_count, tx_sub);
                    if (tx_sub == 2'd2) begin
                        tx_state <= TX_IDLER_ALL;
                        tx_sub <= 2'd0;
                    end else begin
                        tx_sub <= tx_sub + 2'd1;
                    end
                end

                TX_IDLER_ALL: begin
                    tx_byte <= counter19_byte(idler_all, tx_sub);
                    if (tx_sub == 2'd2) begin
                        tx_state <= TX_COIN_ALL;
                        tx_sub <= 2'd0;
                    end else begin
                        tx_sub <= tx_sub + 2'd1;
                    end
                end

                TX_COIN_ALL: begin
                    tx_byte <= counter15_byte(coin_all, tx_sub[0]);
                    if (tx_sub == 2'd1) begin
                        tx_state <= TX_SIGNAL_ALL;
                        tx_sub <= 2'd0;
                    end else begin
                        tx_sub <= 2'd1;
                    end
                end

                TX_SIGNAL_ALL: begin
                    tx_byte <= counter19_byte(signal_all, tx_sub);
                    if (tx_sub == 2'd2) begin
                        tx_state <= TX_SIGNAL_CELL;
                        tx_sub <= 2'd0;
                        tx_cell <= 8'd0;
                    end else begin
                        tx_sub <= tx_sub + 2'd1;
                    end
                end

                TX_SIGNAL_CELL: begin
                    if (tx_sub == 2'd0) begin
                        tx_byte <= {5'd0, signal_cell[tx_cell][10:8]};
                        tx_sub <= 2'd1;
                    end else begin
                        tx_byte <= signal_cell[tx_cell][7:0];
                        tx_sub <= 2'd0;
                        if (tx_cell == 8'd255) begin
                            tx_state <= TX_IDLER_CELL;
                            tx_cell <= 8'd0;
                        end else begin
                            tx_cell <= tx_cell + 8'd1;
                        end
                    end
                end

                TX_IDLER_CELL: begin
                    if (tx_sub == 2'd0) begin
                        tx_byte <= {5'd0, idler_cell[tx_cell][10:8]};
                        tx_sub <= 2'd1;
                    end else begin
                        tx_byte <= idler_cell[tx_cell][7:0];
                        tx_sub <= 2'd0;
                        if (tx_cell == 8'd255) begin
                            tx_state <= TX_COIN_CELL;
                            tx_cell <= 8'd0;
                        end else begin
                            tx_cell <= tx_cell + 8'd1;
                        end
                    end
                end

                TX_COIN_CELL: begin
                    tx_byte <= {1'b0, coin_cell[tx_cell]};
                    if (tx_cell == 8'd255) begin
                        tx_state <= TX_SIGNAL_HIST;
                        tx_hist <= 4'd0;
                    end else begin
                        tx_cell <= tx_cell + 8'd1;
                    end
                end

                TX_SIGNAL_HIST: begin
                    tx_byte <= signal_hist[tx_hist];
                    if (tx_hist == 4'd9) begin
                        tx_state <= TX_IDLER_HIST;
                        tx_hist <= 4'd0;
                    end else begin
                        tx_hist <= tx_hist + 4'd1;
                    end
                end

                TX_IDLER_HIST: begin
                    tx_byte <= idler_hist[tx_hist];
                    if (tx_hist == 4'd9) begin
                        tx_state <= TX_MULTI2_PAIR;
                        tx_hist <= 4'd0;
                        tx_sub <= 2'd0;
                        tx_pair <= 3'd0;
                    end else begin
                        tx_hist <= tx_hist + 4'd1;
                    end
                end

                TX_MULTI2_PAIR: begin
                    case (tx_pair)
                        3'd0: tx_byte <= counter32_byte(multi2_12_count, tx_sub);
                        3'd1: tx_byte <= counter32_byte(multi2_13_count, tx_sub);
                        3'd2: tx_byte <= counter32_byte(multi2_14_count, tx_sub);
                        3'd3: tx_byte <= counter32_byte(multi2_23_count, tx_sub);
                        3'd4: tx_byte <= counter32_byte(multi2_24_count, tx_sub);
                        default: tx_byte <= counter32_byte(multi2_34_count, tx_sub);
                    endcase

                    if (tx_sub == 2'd3) begin
                        tx_sub <= 2'd0;
                        if (tx_pair == 3'd5) begin
                            tx_state <= TX_IDLE;
                            tx_pair <= 3'd0;
                        end else begin
                            tx_pair <= tx_pair + 3'd1;
                        end
                    end else begin
                        tx_sub <= tx_sub + 2'd1;
                    end
                end

                default: begin
                    tx_byte <= 8'd0;
                    tx_state <= TX_IDLE;
                end
            endcase
        end

        if (write_start) begin
            hist_busy <= 1'b1;
            getsignal <= 256'd0;
            getidler <= 256'd0;
        end

        if (Write == 1'b0) begin
            time1 <= 16'd0;
            write_scan <= 1'b0;
            write_cell <= 8'd0;
            write_phase <= 7'd0;
        end else begin
            time1 <= time1 + 16'd1;

            if (time1 == ws_aoddelay) begin
                write_scan <= 1'b1;
                write_cell <= 8'd0;
                write_phase <= 7'd1;
            end else if (write_scan) begin
                if (write_phase == (CELL_PERIOD - 1)) begin
                    write_phase <= 7'd0;
                    if (write_cell == (CELL_COUNT - 1)) begin
                        write_scan <= 1'b0;
                    end else begin
                        write_cell <= write_cell + 8'd1;
                    end
                end else begin
                    write_phase <= write_phase + 7'd1;
                end
            end
        end

        if (Read == 1'b0) begin
            time2 <= 16'd0;
            read_scan <= 1'b0;
            read_cell <= 8'd0;
            read_phase <= 7'd0;
        end else begin
            time2 <= time2 + 16'd1;

            if (time2 == ri_aoddelay) begin
                read_scan <= 1'b1;
                read_cell <= 8'd0;
                read_phase <= 7'd1;
            end else if (read_scan) begin
                if (read_phase == (CELL_PERIOD - 1)) begin
                    read_phase <= 7'd0;
                    if (read_cell == (CELL_COUNT - 1)) begin
                        read_scan <= 1'b0;
                    end else begin
                        read_cell <= read_cell + 8'd1;
                    end
                end else begin
                    read_phase <= read_phase + 7'd1;
                end
            end
        end

        if (Write == 1'b1 && in[0] == 1'b1 &&
            time1 > (ws_aoddelay + cellhist + signals) &&
            time1 < (ws_aoddelay + cellhist + signale)) begin
            for (hist_i = 0; hist_i < HIST_BINS; hist_i = hist_i + 1) begin
                if (time1 > (ws_aoddelay + cellhist + signals + hist_i * sdetectwindow) &&
                    time1 < (ws_aoddelay + cellhist + signals + (hist_i + 1) * sdetectwindow)) begin
                    signal_hist[hist_i] <= signal_hist[hist_i] + 8'd1;
                end
            end
        end

        if (Read == 1'b1 && idler_clicks != 3'd0 &&
            time2 > (ri_aoddelay + cellhist + idlers) &&
            time2 < (ri_aoddelay + cellhist + idlere)) begin
            for (hist_i = 0; hist_i < HIST_BINS; hist_i = hist_i + 1) begin
                if (time2 > (ri_aoddelay + cellhist + idlers + hist_i * idetectwindow) &&
                    time2 < (ri_aoddelay + cellhist + idlers + (hist_i + 1) * idetectwindow)) begin
                    idler_hist[hist_i] <= idler_hist[hist_i] + 8'd1;
                end
            end
        end

        if (signal_hit) begin
            getsignal[write_cell] <= 1'b1;
            signal_all <= signal_all + 19'd1;
        end

        if (Write == 1'b1 && time1 == WRITE_SEQ_TICK) begin
            write_count <= write_count + 19'd1;
        end

        if (idler_hit) begin
            getidler[read_cell] <= 1'b1;
            idler_all <= idler_all + 19'd1;
            if (idler_clicks == 3'd2) begin
                if (in[1] && in[2]) begin
                    multi2_12_count <= multi2_12_count + 32'd1;
                end else if (in[1] && in[3]) begin
                    multi2_13_count <= multi2_13_count + 32'd1;
                end else if (in[1] && in[4]) begin
                    multi2_14_count <= multi2_14_count + 32'd1;
                end else if (in[2] && in[3]) begin
                    multi2_23_count <= multi2_23_count + 32'd1;
                end else if (in[2] && in[4]) begin
                    multi2_24_count <= multi2_24_count + 32'd1;
                end else begin
                    multi2_34_count <= multi2_34_count + 32'd1;
                end
            end
        end

        if (Read == 1'b1 && time2 == READ_AGG_TICK && agg_active == 1'b0) begin
            agg_active <= 1'b1;
            agg_flush <= 1'b0;
            agg_write_valid <= 1'b0;
            agg_done <= 1'b0;
            agg_cell <= 8'd0;
        end

        if (agg_active == 1'b1 || agg_flush == 1'b1) begin
            if (agg_write_valid == 1'b1) begin
                if (agg_signal_hit == 1'b1) begin
                    signal_cell[agg_write_cell] <= agg_signal_old + 11'd1;
                end

                if (agg_idler_hit == 1'b1) begin
                    idler_cell[agg_write_cell] <= agg_idler_old + 11'd1;
                end

                if (agg_signal_hit == 1'b1 && agg_idler_hit == 1'b1) begin
                    coin_cell[agg_write_cell] <= agg_coin_old + 7'd1;
                    coin_all <= coin_all + 15'd1;
                end
            end

            if (agg_active == 1'b1) begin
                agg_signal_old <= signal_cell[agg_cell];
                agg_idler_old <= idler_cell[agg_cell];
                agg_coin_old <= coin_cell[agg_cell];
                agg_signal_hit <= getsignal[agg_cell];
                agg_idler_hit <= getidler[agg_cell];
                agg_write_cell <= agg_cell;
                agg_write_valid <= 1'b1;

                if (agg_cell == (CELL_COUNT - 1)) begin
                    agg_active <= 1'b0;
                    agg_flush <= 1'b1;
                end else begin
                    agg_cell <= agg_cell + 8'd1;
                end
            end else begin
                agg_flush <= 1'b0;
                agg_write_valid <= 1'b0;
                agg_done <= 1'b1;
            end
        end

        if (agg_done == 1'b1) begin
            agg_done <= 1'b0;
            getsignal <= 256'd0;
            getidler <= 256'd0;
            read_count <= read_count + 19'd1;
            hist_busy <= 1'b0;
        end
    end

endmodule
