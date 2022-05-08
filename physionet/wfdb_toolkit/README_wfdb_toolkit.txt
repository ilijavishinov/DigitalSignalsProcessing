First step: Cygwin installation with packages
    - installation of 64-bit cygwin
    - required packages
        - from the Devel category: gcc-core, gcc-fortran, make
        - from the Libs category: libcurl-devel, libexpat-devel
        - from the Net category: curl
        - from the X11 category: xinit, xview-devel

Second step: Installing wfdb toolkit
Commands:
    curl https://physionet.org/physiotools/wfdb.tar.gz | tar xvz"
    cd wfdb-10.m.n
    ./configure
    make install
    make check - making sure wfdb is properly installed
