"""Print the original CUDA training arguments for a registered scale extension."""

import argparse
import shlex

FIELDS = ("capacity", "steps", "seed", "learning-rate", "factor-rank", "warmup-steps")
RECIPES = {
    "xl_joint_32768_s7301": ('C-XL', 65536, 7301, 0.0002, 4, 0),
    "xxl_joint_32768_s7301": ('C-XXL', 32768, 7301, 0.0002, 4, 0),
    "xl_joint_32768_s7302": ('C-XL', 32768, 7302, 0.0002, 4, 0),
    "xxl_joint_32768_s7302": ('C-XXL', 16384, 7302, 0.0002, 4, 0),
    "xl_joint_32768_s7303": ('C-XL', 32768, 7303, 0.0002, 4, 0),
    "xxl_joint_32768_s7303": ('C-XXL', 16384, 7303, 0.0002, 4, 0),
    "xl_lr_low_s7301": ('C-XL', 8192, 7301, 0.0001, 4, 0),
    "xl_lr_high_s7301": ('C-XL', 8192, 7301, 0.0004, 4, 0),
    "xl_warm512_s7301": ('C-XL', 8192, 7301, 0.0002, 4, 512),
    "xl_rank8_s7301": ('C-XL', 8192, 7301, 0.0002, 8, 0),
    "xl_depthmatch_s7301": ('C-XL-D', 8192, 7301, 0.0002, 4, 0),
    "xl_widthmatch_s7301": ('C-XL-W', 8192, 7301, 0.0002, 4, 0),
    "xl_lr_low_s7302": ('C-XL', 8192, 7302, 0.0001, 4, 0),
    "xl_lr_low_s7303": ('C-XL', 8192, 7303, 0.0001, 4, 0),
    "xxl_lr_low_s7301": ('C-XXL', 8192, 7301, 0.0001, 4, 0),
    "xxl_lr_high_s7301": ('C-XXL', 8192, 7301, 0.0004, 4, 0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=RECIPES)
    args = parser.parse_args()
    command = ["--run-id", args.run_id]
    for flag, value in zip(FIELDS, RECIPES[args.run_id]):
        command.extend(["--" + flag, str(value)])
    print(shlex.join(command))


if __name__ == "__main__":
    main()
