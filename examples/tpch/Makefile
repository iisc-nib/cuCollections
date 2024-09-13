build:
	mkdir -p build
	nvcc  -I$(SPACE)/tools/arrow/include/ -L$(SPACE)/tools/arrow/lib q11.cu -lparquet -larrow -diag-suppress 2810 -diag-suppress 611 -o build/q11

clean:
	rm -rf build