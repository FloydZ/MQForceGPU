# n=37
# m=37
K=20
for n in {35..40}; do
	# ab 13 wird die compilezeit unchristlich
	for u in {7..10}; do
		for d in {1..3}; do
			make clean  &> /dev/null
			python3 gen_shared_kernel.py -u ${u} -d ${d} > kernel.inc  
			make UNROLL=${u} MAX_K=${K}  &> /dev/null
			echo "n=${n}, K=${K} UNROLL=${u}, d=${d}"
			python3 gen_sys.py -m ${n} -n ${n} | ./guess 0
			echo "\n\n"
		done
	done
done

