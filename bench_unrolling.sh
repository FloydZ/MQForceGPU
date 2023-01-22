# n=37
# m=37
K=20
for n in {35..40}; do
	# ab 13 wird die compilezeit unchristlich
	for u in {7..10}; do
		for w in {5..10}; do
			make clean  &> /dev/null
			python3 gen_kernel.py -u ${u} -w ${w} > kernel.inc  
			make UNROLL=${u} MAX_K=${K}  &> /dev/null
			echo "n=${n}, K=${K} UNROLL=${u}, w=${w}"
			python3 gen_sys.py -m ${n} -n ${n} | ./guess 0
			echo "\n\n"
		done
	done
done

