#!/bin/bash
for ((i=1;i<=100;i++))
do
    python3.5 main.py --phase test --sm_idx $i
done
