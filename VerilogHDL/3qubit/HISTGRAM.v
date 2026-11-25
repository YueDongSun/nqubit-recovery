module HISTGRAM(clk, in, Write, Read, tx_start, tx_next, tx_byte, hist_busy);

    input clk;
    input [4:0] in;
    input Write, Read;
    input tx_start;
    input tx_next;

    output [7:0] tx_byte;
    output hist_busy;

    parameter signals = 10;
    parameter signale = 30;
    parameter idlers = 24;
    parameter idlere = 44;
    parameter ws_aoddelay = 1080;
    parameter ri_aoddelay = 1130;
    parameter sdetectwindow = (signale - signals) / 10;
    parameter idetectwindow = (idlere - idlers) / 10;
    parameter cellhist = 12000;

    localparam integer BIN_COUNT = 75;
    localparam integer HIST_RAM_DEPTH = 128;
    localparam integer CELL_TICKS = 300;
    localparam integer CELL_BINS = 10;
    localparam integer HIST_TICKS = BIN_COUNT * CELL_TICKS;
    localparam integer FRAME_BYTES = 1044;
    localparam integer WRITE_SEQ_TICK = HIST_TICKS + ws_aoddelay;
    localparam integer READ_AGG_TICK = HIST_TICKS + ri_aoddelay + 100;

    localparam integer TX_MULTI2_OFFSET = 0;
    localparam integer TX_MULTI3_OFFSET = TX_MULTI2_OFFSET + 4;
    localparam integer TX_MULTI4_OFFSET = TX_MULTI3_OFFSET + 4;
    localparam integer TX_MULTI2_PAIR_OFFSET = TX_MULTI4_OFFSET + 4;
    localparam integer TX_READ_COUNT_OFFSET = TX_MULTI2_PAIR_OFFSET + 6 * 4;
    localparam integer TX_WRITE_COUNT_OFFSET = TX_READ_COUNT_OFFSET + 8;
    localparam integer TX_SIGNAL_ALL_OFFSET = TX_WRITE_COUNT_OFFSET + 8;
    localparam integer TX_IDLER_ALL_OFFSET = TX_SIGNAL_ALL_OFFSET + 4;
    localparam integer TX_COIN_ALL_OFFSET = TX_IDLER_ALL_OFFSET + 4;
    localparam integer TX_SIGNAL_BIN_OFFSET = TX_COIN_ALL_OFFSET + 4;
    localparam integer TX_IDLER_BIN_OFFSET = TX_SIGNAL_BIN_OFFSET + BIN_COUNT * 4;
    localparam integer TX_COIN_BIN_OFFSET = TX_IDLER_BIN_OFFSET + BIN_COUNT * 4;
    localparam integer TX_SIGNAL_CELL_OFFSET = TX_COIN_BIN_OFFSET + BIN_COUNT * 4;
    localparam integer TX_IDLER_CELL_OFFSET = TX_SIGNAL_CELL_OFFSET + CELL_BINS * 4;

    localparam TX_IDLE = 4'd0;
    localparam TX_MULTI2 = 4'd1;
    localparam TX_MULTI3 = 4'd2;
    localparam TX_MULTI4 = 4'd3;
    localparam TX_MULTI2_PAIR = 4'd4;
    localparam TX_READ_COUNT = 4'd5;
    localparam TX_WRITE_COUNT = 4'd6;
    localparam TX_SIGNAL_ALL = 4'd7;
    localparam TX_IDLER_ALL = 4'd8;
    localparam TX_COIN_ALL = 4'd9;
    localparam TX_SIGNAL_BIN = 4'd10;
    localparam TX_IDLER_BIN = 4'd11;
    localparam TX_COIN_BIN = 4'd12;
    localparam TX_SIGNAL_CELL = 4'd13;
    localparam TX_IDLER_CELL = 4'd14;

    localparam PREF_SIGNAL_BIN = 3'd0;
    localparam PREF_IDLER_BIN = 3'd1;
    localparam PREF_COIN_BIN = 3'd2;
    localparam PREF_SIGNAL_CELL = 3'd3;
    localparam PREF_IDLER_CELL = 3'd4;

    reg hist_busy = 1'b0;
    reg [7:0] tx_byte = 8'd0;
    reg [3:0] tx_state = TX_IDLE;
    reg [2:0] tx_sub = 3'd0;
    reg [6:0] tx_bin = 7'd0;
    reg [3:0] tx_cell = 4'd0;
    reg [2:0] tx_pair = 3'd0;

    // UART leaves enough clk cycles between accepted bytes.
    // The formatter below prepares the byte for the current tx_state/tx_sub.
    //
    // Important:
    // Do not pre-emit the next field's byte0 in the same tx_next cycle that
    // emits the current field's last byte. The UART samples tx_byte for the
    // current byte; assigning tx_byte again to the next field byte corrupts
    // byte boundaries. State transitions only change tx_state/tx_sub.
    reg [6:0] tx_read_bin = 7'd0;
    reg [3:0] tx_read_cell = 4'd0;
    reg [2:0] tx_read_source = PREF_SIGNAL_BIN;
    reg [1:0] tx_read_sel = 2'd0;
    reg [31:0] tx_signal_word = 32'd0;
    reg [31:0] tx_idler_word = 32'd0;
    reg [31:0] tx_coin_word = 32'd0;
    reg [31:0] tx_signal_cell_word = 32'd0;
    reg [31:0] tx_idler_cell_word = 32'd0;
    reg [7:0] tx_prefetch_byte = 8'd0;

    reg [31:0] multi2_count = 32'd0;
    reg [31:0] multi2_12_count = 32'd0;
    reg [31:0] multi2_13_count = 32'd0;
    reg [31:0] multi2_14_count = 32'd0;
    reg [31:0] multi2_23_count = 32'd0;
    reg [31:0] multi2_24_count = 32'd0;
    reg [31:0] multi2_34_count = 32'd0;
    reg [31:0] multi3_count = 32'd0;
    reg [31:0] multi4_count = 32'd0;
    reg [63:0] read_count = 64'd0;
    reg [63:0] write_count = 64'd0;
    reg [31:0] signal_all = 32'd0;
    reg [31:0] idler_all = 32'd0;
    reg [31:0] coin_all = 32'd0;

    // Cyclone III implements a 32-bit-wide M9K at a physical depth of 128.
    // Only entries 0..74 are functional; padding the unused tail keeps the
    // initialization image aligned with the physical RAM geometry.
    (* ramstyle = "M9K" *) reg [31:0] signal_hist [0:HIST_RAM_DEPTH-1];
    (* ramstyle = "M9K" *) reg [31:0] idler_hist [0:HIST_RAM_DEPTH-1];
    (* ramstyle = "M9K" *) reg [31:0] coin_hist [0:HIST_RAM_DEPTH-1];

    reg [31:0] signal_cell [0:CELL_BINS-1];
    reg [31:0] idler_cell [0:CELL_BINS-1];

    reg [74:0] getsignal = 75'd0;
    reg [74:0] getidler = 75'd0;

    reg [19:0] time1 = 20'd0;
    reg [19:0] time2 = 20'd0;
    reg Write_d = 1'b0;
    reg [8:0] write_phase = 9'd0;
    reg [8:0] read_phase = 9'd0;
    reg [6:0] write_bin = 7'd0;
    reg [6:0] read_bin = 7'd0;

    reg agg_active = 1'b0;
    reg agg_flush = 1'b0;
    reg agg_write_valid = 1'b0;
    reg agg_done = 1'b0;
    reg [6:0] agg_bin = 7'd0;
    reg [6:0] agg_write_bin = 7'd0;
    reg agg_signal_hit = 1'b0;
    reg agg_idler_hit = 1'b0;
    reg [31:0] agg_signal_old = 32'd0;
    reg [31:0] agg_idler_old = 32'd0;
    reg [31:0] agg_coin_old = 32'd0;

    wire [2:0] idler_clicks;
    wire write_start = Write & ~Write_d;

    wire signal_hit = (Write == 1'b1 && in[0] == 1'b1 &&
                       time1 > ws_aoddelay && time1 < (ws_aoddelay + HIST_TICKS + 1) &&
                       write_bin < BIN_COUNT &&
                       write_phase > signals && write_phase < signale);

    wire idler_hit = (Read == 1'b1 && idler_clicks != 3'd0 &&
                      time2 > ri_aoddelay && time2 < (ri_aoddelay + HIST_TICKS + 1) &&
                      read_bin < BIN_COUNT &&
                      read_phase > idlers && read_phase < idlere);

    assign idler_clicks = {2'b0, in[1]} + {2'b0, in[2]} +
                          {2'b0, in[3]} + {2'b0, in[4]};

    integer init_i;
    integer cell_i;

    initial begin
        for (init_i = 0; init_i < HIST_RAM_DEPTH; init_i = init_i + 1) begin
            signal_hist[init_i] = 32'd0;
            idler_hist[init_i] = 32'd0;
            coin_hist[init_i] = 32'd0;
        end

        for (init_i = 0; init_i < CELL_BINS; init_i = init_i + 1) begin
            signal_cell[init_i] = 32'd0;
            idler_cell[init_i] = 32'd0;
        end
    end

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

    function [7:0] counter64_byte;
        input [63:0] value;
        input [2:0] byte_sel;
        begin
            case (byte_sel)
                3'd0: counter64_byte = value[63:56];
                3'd1: counter64_byte = value[55:48];
                3'd2: counter64_byte = value[47:40];
                3'd3: counter64_byte = value[39:32];
                3'd4: counter64_byte = value[31:24];
                3'd5: counter64_byte = value[23:16];
                3'd6: counter64_byte = value[15:8];
                default: counter64_byte = value[7:0];
            endcase
        end
    endfunction

    // Pipelined RAM/cell byte formatter.
    // It prepares the byte corresponding to the current tx_state/tx_sub.
    // There is enough UART inter-byte idle time for tx_prefetch_byte to settle
    // before the next tx_next pulse.
    always @(posedge clk) begin
        case (tx_state)
            TX_SIGNAL_BIN: begin
                tx_read_bin <= tx_bin;
                tx_read_source <= PREF_SIGNAL_BIN;
                tx_read_sel <= tx_sub[1:0];
            end

            TX_IDLER_BIN: begin
                tx_read_bin <= tx_bin;
                tx_read_source <= PREF_IDLER_BIN;
                tx_read_sel <= tx_sub[1:0];
            end

            TX_COIN_BIN: begin
                tx_read_bin <= tx_bin;
                tx_read_source <= PREF_COIN_BIN;
                tx_read_sel <= tx_sub[1:0];
            end

            TX_SIGNAL_CELL: begin
                tx_read_cell <= tx_cell;
                tx_read_source <= PREF_SIGNAL_CELL;
                tx_read_sel <= tx_sub[1:0];
            end

            TX_IDLER_CELL: begin
                tx_read_cell <= tx_cell;
                tx_read_source <= PREF_IDLER_CELL;
                tx_read_sel <= tx_sub[1:0];
            end

            default: begin
                tx_read_sel <= tx_read_sel;
            end
        endcase

        tx_signal_word <= signal_hist[tx_read_bin];
        tx_idler_word <= idler_hist[tx_read_bin];
        tx_coin_word <= coin_hist[tx_read_bin];
        tx_signal_cell_word <= signal_cell[tx_read_cell];
        tx_idler_cell_word <= idler_cell[tx_read_cell];

        case (tx_read_source)
            PREF_SIGNAL_BIN:
                tx_prefetch_byte <= counter32_byte(tx_signal_word, tx_read_sel);

            PREF_IDLER_BIN:
                tx_prefetch_byte <= counter32_byte(tx_idler_word, tx_read_sel);

            PREF_COIN_BIN:
                tx_prefetch_byte <= counter32_byte(tx_coin_word, tx_read_sel);

            PREF_SIGNAL_CELL:
                tx_prefetch_byte <= counter32_byte(tx_signal_cell_word, tx_read_sel);

            default:
                tx_prefetch_byte <= counter32_byte(tx_idler_cell_word, tx_read_sel);
        endcase
    end

    always @(posedge clk) begin
        Write_d <= Write;

        if (tx_start) begin
            tx_state <= TX_MULTI2;
            tx_sub <= 3'd1;
            tx_bin <= 7'd0;
            tx_cell <= 4'd0;
            tx_pair <= 3'd0;
            tx_byte <= counter32_byte(multi2_count, 2'd0);

        end else if (tx_next) begin
            case (tx_state)
                TX_MULTI2: begin
                    tx_byte <= counter32_byte(multi2_count, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_MULTI3;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_MULTI3: begin
                    tx_byte <= counter32_byte(multi3_count, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_MULTI4;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_MULTI4: begin
                    tx_byte <= counter32_byte(multi4_count, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_MULTI2_PAIR;
                        tx_pair <= 3'd0;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_MULTI2_PAIR: begin
                    case (tx_pair)
                        3'd0: tx_byte <= counter32_byte(multi2_12_count, tx_sub[1:0]);
                        3'd1: tx_byte <= counter32_byte(multi2_13_count, tx_sub[1:0]);
                        3'd2: tx_byte <= counter32_byte(multi2_14_count, tx_sub[1:0]);
                        3'd3: tx_byte <= counter32_byte(multi2_23_count, tx_sub[1:0]);
                        3'd4: tx_byte <= counter32_byte(multi2_24_count, tx_sub[1:0]);
                        default: tx_byte <= counter32_byte(multi2_34_count, tx_sub[1:0]);
                    endcase

                    if (tx_sub == 3'd3) begin
                        tx_sub <= 3'd0;
                        if (tx_pair == 3'd5) begin
                            tx_pair <= 3'd0;
                            tx_state <= TX_READ_COUNT;
                        end else begin
                            tx_pair <= tx_pair + 3'd1;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_READ_COUNT: begin
                    tx_byte <= counter64_byte(read_count, tx_sub);
                    if (tx_sub == 3'd7) begin
                        tx_state <= TX_WRITE_COUNT;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_WRITE_COUNT: begin
                    tx_byte <= counter64_byte(write_count, tx_sub);
                    if (tx_sub == 3'd7) begin
                        tx_state <= TX_SIGNAL_ALL;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_SIGNAL_ALL: begin
                    tx_byte <= counter32_byte(signal_all, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_IDLER_ALL;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_IDLER_ALL: begin
                    tx_byte <= counter32_byte(idler_all, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_COIN_ALL;
                        tx_sub <= 3'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_COIN_ALL: begin
                    tx_byte <= counter32_byte(coin_all, tx_sub[1:0]);
                    if (tx_sub == 3'd3) begin
                        tx_state <= TX_SIGNAL_BIN;
                        tx_sub <= 3'd0;
                        tx_bin <= 7'd0;
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_SIGNAL_BIN: begin
                    tx_byte <= tx_prefetch_byte;
                    if (tx_sub == 3'd3) begin
                        if (tx_bin == (BIN_COUNT - 1)) begin
                            tx_state <= TX_IDLER_BIN;
                            tx_bin <= 7'd0;
                            tx_sub <= 3'd0;
                        end else begin
                            tx_bin <= tx_bin + 7'd1;
                            tx_sub <= 3'd0;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_IDLER_BIN: begin
                    tx_byte <= tx_prefetch_byte;
                    if (tx_sub == 3'd3) begin
                        if (tx_bin == (BIN_COUNT - 1)) begin
                            tx_state <= TX_COIN_BIN;
                            tx_bin <= 7'd0;
                            tx_sub <= 3'd0;
                        end else begin
                            tx_bin <= tx_bin + 7'd1;
                            tx_sub <= 3'd0;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_COIN_BIN: begin
                    tx_byte <= tx_prefetch_byte;
                    if (tx_sub == 3'd3) begin
                        if (tx_bin == (BIN_COUNT - 1)) begin
                            tx_state <= TX_SIGNAL_CELL;
                            tx_cell <= 4'd0;
                            tx_sub <= 3'd0;
                        end else begin
                            tx_bin <= tx_bin + 7'd1;
                            tx_sub <= 3'd0;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_SIGNAL_CELL: begin
                    tx_byte <= tx_prefetch_byte;
                    if (tx_sub == 3'd3) begin
                        if (tx_cell == 4'd9) begin
                            tx_state <= TX_IDLER_CELL;
                            tx_cell <= 4'd0;
                            tx_sub <= 3'd0;
                        end else begin
                            tx_cell <= tx_cell + 4'd1;
                            tx_sub <= 3'd0;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                TX_IDLER_CELL: begin
                    tx_byte <= tx_prefetch_byte;
                    if (tx_sub == 3'd3) begin
                        if (tx_cell == 4'd9) begin
                            tx_state <= TX_IDLE;
                            tx_cell <= 4'd0;
                            tx_sub <= 3'd0;
                        end else begin
                            tx_cell <= tx_cell + 4'd1;
                            tx_sub <= 3'd0;
                        end
                    end else begin
                        tx_sub <= tx_sub + 3'd1;
                    end
                end

                default: begin
                    tx_byte <= 8'd0;
                    tx_state <= TX_IDLE;
                    tx_sub <= 3'd0;
                end
            endcase
        end

        if (write_start) begin
            hist_busy <= 1'b1;
            getsignal <= 75'd0;
            getidler <= 75'd0;
        end

        if (Write == 1'b0) begin
            time1 <= 20'd0;
            write_phase <= 9'd0;
            write_bin <= 7'd0;
        end else begin
            time1 <= time1 + 20'd1;

            if (time1 < ws_aoddelay) begin
                write_phase <= 9'd0;
                write_bin <= 7'd0;
            end else if (time1 == ws_aoddelay) begin
                write_phase <= 9'd1;
                write_bin <= 7'd0;
            end else if (time1 < (ws_aoddelay + HIST_TICKS)) begin
                if (write_phase == (CELL_TICKS - 1)) begin
                    write_phase <= 9'd0;
                    write_bin <= write_bin + 7'd1;
                end else begin
                    write_phase <= write_phase + 9'd1;
                end
            end
        end

        if (Read == 1'b0) begin
            time2 <= 20'd0;
            read_phase <= 9'd0;
            read_bin <= 7'd0;
        end else begin
            time2 <= time2 + 20'd1;

            if (time2 < ri_aoddelay) begin
                read_phase <= 9'd0;
                read_bin <= 7'd0;
            end else if (time2 == ri_aoddelay) begin
                read_phase <= 9'd1;
                read_bin <= 7'd0;
            end else if (time2 < (ri_aoddelay + HIST_TICKS)) begin
                if (read_phase == (CELL_TICKS - 1)) begin
                    read_phase <= 9'd0;
                    read_bin <= read_bin + 7'd1;
                end else begin
                    read_phase <= read_phase + 9'd1;
                end
            end
        end

        if (signal_hit) begin
            getsignal[write_bin] <= 1'b1;
        end

        if (Write == 1'b1 && time1 == WRITE_SEQ_TICK) begin
            write_count <= write_count + 64'd1;
        end

        if (idler_hit) begin
            case (idler_clicks)
                3'd1: begin
                    getidler[read_bin] <= 1'b1;
                end

                3'd2: begin
                    multi2_count <= multi2_count + 32'd1;
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

                3'd3: begin
                    multi3_count <= multi3_count + 32'd1;
                end

                default: begin
                    multi4_count <= multi4_count + 32'd1;
                end
            endcase
        end

        if (Write == 1'b1 && in[0] == 1'b1 &&
            time1 > (ws_aoddelay + cellhist + signals) &&
            time1 < (ws_aoddelay + cellhist + signale)) begin
            for (cell_i = 0; cell_i < CELL_BINS; cell_i = cell_i + 1) begin
                if (time1 > (ws_aoddelay + cellhist + signals + cell_i * sdetectwindow) &&
                    time1 < (ws_aoddelay + cellhist + signals + (cell_i + 1) * sdetectwindow)) begin
                    signal_cell[cell_i] <= signal_cell[cell_i] + 32'd1;
                end
            end
        end

        if (Read == 1'b1 && idler_clicks != 3'd0 &&
            time2 > (ri_aoddelay + cellhist + idlers) &&
            time2 < (ri_aoddelay + cellhist + idlere)) begin
            for (cell_i = 0; cell_i < CELL_BINS; cell_i = cell_i + 1) begin
                if (time2 > (ri_aoddelay + cellhist + idlers + cell_i * idetectwindow) &&
                    time2 < (ri_aoddelay + cellhist + idlers + (cell_i + 1) * idetectwindow)) begin
                    idler_cell[cell_i] <= idler_cell[cell_i] + 32'd1;
                end
            end
        end

        if (Read == 1'b1 && time2 == READ_AGG_TICK && agg_active == 1'b0) begin
            agg_active <= 1'b1;
            agg_flush <= 1'b0;
            agg_write_valid <= 1'b0;
            agg_done <= 1'b0;
            agg_bin <= 7'd0;
        end

        if (agg_active == 1'b1 || agg_flush == 1'b1) begin
            if (agg_write_valid == 1'b1) begin
                if (agg_signal_hit == 1'b1) begin
                    signal_hist[agg_write_bin] <= agg_signal_old + 32'd1;
                    signal_all <= signal_all + 32'd1;
                end

                if (agg_idler_hit == 1'b1) begin
                    idler_hist[agg_write_bin] <= agg_idler_old + 32'd1;
                    idler_all <= idler_all + 32'd1;
                end

                if (agg_signal_hit == 1'b1 && agg_idler_hit == 1'b1) begin
                    coin_hist[agg_write_bin] <= agg_coin_old + 32'd1;
                    coin_all <= coin_all + 32'd1;
                end
            end

            if (agg_active == 1'b1) begin
                agg_signal_old <= signal_hist[agg_bin];
                agg_idler_old <= idler_hist[agg_bin];
                agg_coin_old <= coin_hist[agg_bin];

                agg_signal_hit <= getsignal[agg_bin];
                agg_idler_hit <= getidler[agg_bin];

                agg_write_bin <= agg_bin;
                agg_write_valid <= 1'b1;

                if (agg_bin == (BIN_COUNT - 1)) begin
                    agg_active <= 1'b0;
                    agg_flush <= 1'b1;
                end else begin
                    agg_bin <= agg_bin + 7'd1;
                end
            end else begin
                agg_flush <= 1'b0;
                agg_write_valid <= 1'b0;
                agg_done <= 1'b1;
            end
        end

        if (agg_done == 1'b1) begin
            agg_done <= 1'b0;
            getsignal <= 75'd0;
            getidler <= 75'd0;
            read_count <= read_count + 64'd1;
            hist_busy <= 1'b0;
        end
    end

endmodule
