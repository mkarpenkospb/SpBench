echo "Run benchmark for targets: $SPBENCH_TARGETS"

# For each target we run as separate process for each matrix
cat $SPBENCH_TARGETS | while read target; do
  cat data/config_kronecker.txt | while read test; do
      # Ignore lines, which start from comment mark
      if [[ ${test::1} != "%" ]]; then
        echo "Exec command: ./$target $test"
        ./$target -E $test
      fi
    done
done