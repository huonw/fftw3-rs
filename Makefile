include rust.mk

RUSTC ?= rustc
RUSTFLAGS ?= -O

.PHONY : all
all : fftw examples

.PHONY : check
check : check-fftw

examples: examples/basic

examples/basic: fftw examples/basic.rs
	$(RUSTC) $(RUSTFLAGS) examples/basic.rs -L build --out-dir build

$(eval $(call RUST_CRATE, src/, build/))
