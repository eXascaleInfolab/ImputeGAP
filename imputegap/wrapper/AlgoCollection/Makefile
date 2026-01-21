# need to install (unix):
# sudo apt install build-essential cmake gfortran
# sudo apt install libarmadillo-dev
# sudo apt-get install libboost-all-dev
# sudo apt-get install libmlpack-dev

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# export CPLUS_INCLUDE_PATH=/usr/include/mlpack:$CPLUS_INCLUDE_PATH

# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt-get update
# sudo apt-get install --only-upgrade libstdc++6

# make libSTMVL.dylib libIterativeSVD.dylib libGROUSE.dylib libDynaMMo.dylib libROSL.dylib libSoftImpute.dylib libSPIRIT.dylib libSVT.dylib libTKCM.dylib libCDREC.dylib
# # make libSTMVL.so libIterativeSVD.so libGROUSE.so libDynaMMo.so libROSL.so libSoftImpute.so libSPIRIT.so libSVT.so libTKCM.so libCDREC.so

all: libAlgoCollection.so AlgoCollection.dll
# ==================================================================================================================================

libAlgoCollection.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o libAlgoCollection.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 Algebra/Auxiliary.cpp \
	Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp \
	Algorithms/NMFMissingValueRecovery.cpp Algorithms/DynaMMo.cpp Algorithms/SVT.cpp Algorithms/ROSL.cpp \
	Algorithms/IterativeSVD.cpp Algorithms/SoftImpute.cpp Algorithms/CDMissingValueRecovery.cpp \
	Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp shared/SharedLibFunctions.cpp \
	-lopenblas -larpack -lmlpack

# ==================================================================================================================================

libSTMVL.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_stmvl.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algorithms/ST_MVL.cpp shared/SharedSTMVL.cpp \
	-lopenblas -larpack -lmlpack

libSTMVL.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_stmvl_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algorithms/ST_MVL.cpp shared/SharedSTMVL.cpp \
	-larmadillo -lopenblas -larpack
# ==================================================================================================================================

libIterativeSVD.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_iterative_svd.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algebra/RSVD.cpp Algorithms/IterativeSVD.cpp shared/SharedLibIterativeSVD.cpp \
	-lopenblas -larpack

libIterativeSVD.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_iterative_svd_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algebra/RSVD.cpp Algorithms/IterativeSVD.cpp shared/SharedLibIterativeSVD.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libGROUSE.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_grouse.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algorithms/GROUSE.cpp shared/SharedLibGROUSE.cpp \
	-lopenblas -larpack

libGROUSE.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_grouse_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algorithms/GROUSE.cpp shared/SharedLibGROUSE.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================


libDynaMMo.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_dynammo.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algebra/Auxiliary.cpp Algorithms/DynaMMo.cpp shared/SharedLibDynaMMo.cpp \
	-lopenblas -larpack

libDynaMMo.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_dynammo_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algebra/Auxiliary.cpp Algorithms/DynaMMo.cpp shared/SharedLibDynaMMo.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libROSL.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_rosl.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algebra/Auxiliary.cpp Algorithms/ROSL.cpp shared/SharedLibROSL.cpp \
	-lopenblas -larpack

libROSL.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_rosl_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algebra/Auxiliary.cpp Algorithms/ROSL.cpp shared/SharedLibROSL.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libSoftImpute.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_soft_impute.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algebra/RSVD.cpp Algorithms/SoftImpute.cpp shared/SharedLibSoftImpute.cpp \
	-lopenblas -larpack

libSoftImpute.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_soft_impute_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algebra/RSVD.cpp Algorithms/SoftImpute.cpp shared/SharedLibSoftImpute.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libSPIRIT.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_spirit.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algorithms/SPIRIT.cpp shared/SharedLibSPIRIT.cpp \
	-lopenblas -larpack

libSPIRIT.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_spirit_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algorithms/SPIRIT.cpp shared/SharedLibSPIRIT.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libSVT.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_svt.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algorithms/SVT.cpp shared/SharedLibSVT.cpp \
	-lopenblas -larpack

libSVT.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_svt_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algorithms/SVT.cpp shared/SharedLibSVT.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================

libTKCM.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_tkcm.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Algorithms/TKCM.cpp shared/SharedLibTKCM.cpp \
	-lopenblas -larpack

libTKCM.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_tkcm_new.dylib \
	-I/opt/homebrew/include \
	-L/opt/homebrew/lib -L/opt/homebrew/opt/openblas/lib \
	Algorithms/TKCM.cpp shared/SharedLibTKCM.cpp \
	-larmadillo -lopenblas -larpack

# ==================================================================================================================================
libCDREC.so:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o lib_cdrec.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 \
	Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp  Algebra/Auxiliary.cpp \
	Algebra/CentroidDecomposition.cpp  shared/SharedLibCDREC.cpp \
	-lopenblas -larpack -lmlpack
	
libCDREC.dylib:
	clang++ -dynamiclib -O3 -fPIC -std=c++17 -o lib_cdrec_new.dylib \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -L/opt/homebrew/opt/openblas/lib \
    Stats/Correlation.cpp Algorithms/CDMissingValueRecovery.cpp Algebra/Auxiliary.cpp \
    Algebra/CentroidDecomposition.cpp shared/SharedLibCDREC.cpp \
    -larmadillo -lopenblas -larpack
# ==================================================================================================================================



libAlgoCollection.dll:
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -shared -o libAlgoCollection.dll -Wall -Werror -Wextra -pedantic -Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 -fPIC \
	Algebra/Auxiliary.cpp Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp Algorithms/NMFMissingValueRecovery.cpp Algorithms/DynaMMo.cpp Algorithms/SVT.cpp Algorithms/ROSL.cpp \
	Algorithms/IterativeSVD.cpp Algorithms/SoftImpute.cpp Algorithms/CDMissingValueRecovery.cpp Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp shared/SharedLibFunctions.cpp \
	-lopenblas -larpack -lmlpack -m64

nomlpack: libAlgoCollection.so
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o libAlgoCollection.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 Algebra/Auxiliary.cpp \
	Algorithms/TKCM.cpp Algorithms/ST_MVL.cpp Algorithms/SPIRIT.cpp Algorithms/GROUSE.cpp Algorithms/DynaMMo.cpp \
	Algorithms/SVT.cpp Algorithms/ROSL.cpp Algorithms/IterativeSVD.cpp Algorithms/SoftImpute.cpp \
	Algorithms/CDMissingValueRecovery.cpp Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp Stats/Correlation.cpp \
	shared/SharedLibFunctions.cpp -lopenblas -larpack

brian: libAlgoCollection.so
	g++ -O3 -D ARMA_DONT_USE_WRAPPER -fPIC -rdynamic -shared -o libAlgoCollection.so -Wall -Werror -Wextra -pedantic \
	-Wconversion -Wsign-conversion -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -std=gnu++14 Algebra/Auxiliary.cpp \
	Algorithms/ST_MVL.cpp Algorithms/CDMissingValueRecovery.cpp Algebra/CentroidDecomposition.cpp Algebra/RSVD.cpp \
	Stats/Correlation.cpp shared/SharedLibFunctions.cpp -lopenblas -larpack

clean:
	rm -f libAlgoCollection.so libAlgoCollection.dll libCDREC.so libSTMVL.so

clear:
	rm -f libAlgoCollection.so libAlgoCollection.dll libCDREC.so libSTMVL.so


