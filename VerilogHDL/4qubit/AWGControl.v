`timescale 1ns/1ps

module AWGControl(
    input  wire        clk,
    input  wire        triggerall,
    input  wire        hist_busy,
    output reg         Write ,
    output reg         Read ,
    output reg         PumpingW = 1'b0,
    output reg         PumpingR = 1'b0,
    //output reg         freeze_counts = 1'b0,
    output reg         Write_trigger = 1'b0
    //input  wire        hist_updating,
    //output reg [31:0]  timeall = 0
);

    // ---------------- Derived ticks ----------------
    localparam  PUMP_TICKS       = 5020;        // pumping time
    localparam  WRITE_TICKS      = 1_225_00+1045+10;       // write_trial time_all
    localparam  READ_DELAY_TICKS = 4092 ;       	 // storage time
    localparam  READ_TICKS       = 1_225_00+1055+500;       // read_trial time_all
    localparam  PAUSE_TICKS      = 1200;        // intervasl time
    localparam  TOTAL_TICKS      = 1_300_00;       // exp_alltime
    localparam  TRIG_PULSE_TICKS = 1000;         // write&read_trig_time

    // ---------------- State machine ----------------
    reg [2:0] state = 3'd0;
    reg [17:0] counter = 0;
    reg [17:0] total_counter = 0;
    reg [9:0] pulse_counter = 0;

    // ---------------- External trigger synchronizer ----------------
    reg [1:0] trigger_sync = 2'b00;
    wire trigger_active;
    assign trigger_active = trigger_sync[1];

    localparam S_IDLE   = 3'd0;
    localparam S_PUMP   = 3'd1;
    localparam S_WRITE  = 3'd2;
    localparam S_FREEZE = 3'd3;
    localparam S_DONE   = 3'd4;

    // ---------------- Sequential logic ----------------
    always @(posedge clk) begin
        trigger_sync <= {trigger_sync[0], triggerall};

        // total time counter
        if (state != S_IDLE && state != S_DONE) begin
            //timeall <= timeall + 1;
            total_counter <= total_counter + 1;
        end

        case(state)
            S_IDLE: begin
                Write <= 0;
                Read  <= 0;
                PumpingR <= 0;
                //freeze_counts <= 0;
                counter <= 0;
                pulse_counter <= 0;
                Write_trigger <= 0;
                total_counter <= 0;

                // 上升沿触发开始实验序列
                if (trigger_active && !hist_busy) begin
                    state <= S_PUMP;
                    //timeall <= 0;
                    PumpingW <= 1;
						  counter <=0;
                end
					 else begin
						PumpingW <= 0;
					end
            end

            S_PUMP: begin
                if (counter >= (PUMP_TICKS-20)) begin
                    PumpingW <= 0;
                    counter <= 0;
                    Write <= 1;
                    state <= S_WRITE;
                    pulse_counter <= 0;
                    Write_trigger <= 1;
                end
					 else begin
							counter <= counter + 1;
							PumpingW <= 1;
					 end
            end

            S_WRITE: begin

                // 1us external pulse
                if (pulse_counter < TRIG_PULSE_TICKS) begin
                    pulse_counter <= pulse_counter + 1;
                    Write_trigger <= 1;
                end else begin
                    Write_trigger <= 0;
                end

                // start read after READ_DELAY
                if (counter == READ_DELAY_TICKS) begin
                    Read <= 1;
                    //PumpingR <= 1;
                end

                // end write after WRITE_TIME
                if (counter >= (WRITE_TICKS -1)) begin
                    Write <= 0;
                end

                // end read after READ_DELAY + READ_TIME
                if (counter >= (READ_DELAY_TICKS + READ_TICKS -1)) begin
                    Read  <= 0;
                    //PumpingR <= 0;
                    counter <= 0;
                    //freeze_counts <= 1;
                    state <= S_FREEZE;
                end
					 else begin
							counter <= counter + 1;
					 end
            end

            S_FREEZE: begin
                if ((counter >= (PAUSE_TICKS - 10)) && !hist_busy) begin
                    //freeze_counts <= 0;
                    counter <= 0;
                    if (total_counter >= (TOTAL_TICKS)) begin
                        state <= S_DONE;
                    end else begin
                        PumpingW <= 1;
                        state <= S_PUMP;
                        pulse_counter <= 0;
                        Write_trigger <= 0;
                    end
                end
					 else begin
							counter <= counter + 1;
					 end
            end

            S_DONE: begin
                Write <= 0;
                Read  <= 0;
                PumpingW <= 0;
                PumpingR <= 0;
                //freeze_counts <= 1;
                Write_trigger <= 0;

                state <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
endmodule
