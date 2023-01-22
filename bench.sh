for n in {30..50}; do
for K in {19..22}; do
for u in {6..10}; do
for d in {1..10}; do
        make clean  &> /dev/null
        python3 gen_shared_kernel.py -u ${u} -d ${d} > kernel_opt_shared.cuh
        make UNROLL=${u} MAX_K=${K}  &> /dev/null
        echo "n=${n} K=${K} UNROLL=${u} d=${d}"
        python3 gen_sys.py -m ${n} -n ${n} | ./guess 0
        echo "\n\n"
done
done
done
done
