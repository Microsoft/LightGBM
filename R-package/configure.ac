### configure.ac                    -*- Autoconf -*-
# Template used by Autoconf to generate 'configure' script. For more see:
#   * https://unconj.ca/blog/an-autoconf-primer-for-r-package-authors.html
#   * https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup

AC_PREREQ(2.69)
AC_INIT([lightgbm], [4.7.0.99], [], [lightgbm], [])

###########################
# find compiler and flags #
###########################

AC_MSG_CHECKING([location of R])
AC_MSG_RESULT([${R_HOME}])

# set up CPP flags
# find the compiler and compiler flags used by R.
: ${R_HOME=`R HOME`}
if test -z "${R_HOME}"; then
    echo "could not determine R_HOME"
    exit 1
fi
CXX17=`"${R_HOME}/bin/R" CMD config CXX17`
CXX17STD=`"${R_HOME}/bin/R" CMD config CXX17STD`
CXX="${CXX17} ${CXX17STD}"
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXX17FLAGS`
LDFLAGS=`"${R_HOME}/bin/R" CMD config LDFLAGS`
AC_LANG(C++)

# LightGBM-specific flags
LGB_CPPFLAGS=""

#########
# Eigen #
#########

LGB_CPPFLAGS="${LGB_CPPFLAGS} -DEIGEN_MPL2_ONLY -DEIGEN_DONT_PARALLELIZE"

###############
# MM_PREFETCH #
###############

AC_MSG_CHECKING([whether MM_PREFETCH works])
ac_mmprefetch=no
AC_LANG_CONFTEST(
    [
        AC_LANG_PROGRAM(
            [[
                #include <xmmintrin.h>
            ]],
            [[
                int a = 0;
                _mm_prefetch(&a, _MM_HINT_NTA);
                return 0;
            ]]
        )
    ]
)
${CXX} ${CPPFLAGS} ${CXXFLAGS} -o conftest conftest.cpp 2>/dev/null && ./conftest && ac_mmprefetch=yes
AC_MSG_RESULT([${ac_mmprefetch}])
if test "${ac_mmprefetch}" = yes; then
    LGB_CPPFLAGS="${LGB_CPPFLAGS} -DMM_PREFETCH=1"
fi

############
# MM_ALLOC #
############

AC_MSG_CHECKING([whether MM_MALLOC works])
ac_mm_malloc=no
AC_LANG_CONFTEST(
    [
        AC_LANG_PROGRAM(
            [[
                #include <mm_malloc.h>
            ]],
            [[
                char *a = (char*)_mm_malloc(8, 16);
                _mm_free(a);
                return 0;
            ]]
        )
    ]
)
${CXX} ${CPPFLAGS} ${CXXFLAGS} -o conftest conftest.cpp 2>/dev/null && ./conftest && ac_mm_malloc=yes
AC_MSG_RESULT([${ac_mm_malloc}])
if test "${ac_mm_malloc}" = yes; then
    LGB_CPPFLAGS="${LGB_CPPFLAGS} -DMM_MALLOC=1"
fi

##########
# OpenMP #
##########

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
    OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
    OPENMP_CXXFLAGS='-Xclang -fopenmp'
    OPENMP_LIB='-lomp'

    # libomp 15.0+ from brew is keg-only (i.e. not symlinked into the standard paths search by the linker),
    # so need to search in other locations.
    # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
    #
    # If Homebrew is found and libomp was installed with it, this code adds the necessary
    # flags for the compiler to find libomp headers and for the linker to find libomp.dylib.
    HOMEBREW_LIBOMP_PREFIX=""
    if command -v brew >/dev/null 2>&1; then
        ac_brew_openmp=no
        AC_MSG_CHECKING([whether OpenMP was installed via Homebrew])
        brew --prefix libomp >/dev/null 2>&1 && ac_brew_openmp=yes
        AC_MSG_RESULT([${ac_brew_openmp}])
        if test "${ac_brew_openmp}" = yes; then
            HOMEBREW_LIBOMP_PREFIX=`brew --prefix libomp`
            OPENMP_CXXFLAGS="${OPENMP_CXXFLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include"
            OPENMP_LIB="${OPENMP_LIB} -L${HOMEBREW_LIBOMP_PREFIX}/lib"
        fi
    fi
    ac_pkg_openmp=no
    AC_MSG_CHECKING([whether OpenMP will work in a package])
    AC_LANG_CONFTEST(
        [
            AC_LANG_PROGRAM(
                [[
                    #include <omp.h>
                ]],
                [[
                    return (omp_get_max_threads() <= 1);
                ]]
            )
        ]
    )
    ${CXX} ${CPPFLAGS} ${CXXFLAGS} ${LDFLAGS} ${OPENMP_CXXFLAGS} ${OPENMP_LIB} -o conftest conftest.cpp 2>/dev/null && ./conftest && ac_pkg_openmp=yes

    # -Xclang is not portable (it is clang-specific)
    # if compilation above failed, try without that flag
    if test "${ac_pkg_openmp}" = no; then
        if test -f "./conftest"; then
            rm ./conftest
        fi
        OPENMP_CXXFLAGS="-fopenmp"
        ${CXX} ${CPPFLAGS} ${CXXFLAGS} ${LDFLAGS} ${OPENMP_CXXFLAGS} ${OPENMP_LIB} -o conftest conftest.cpp 2>/dev/null && ./conftest && ac_pkg_openmp=yes
    fi

    AC_MSG_RESULT([${ac_pkg_openmp}])
    if test "${ac_pkg_openmp}" = no; then
        OPENMP_CXXFLAGS=''
        OPENMP_LIB=''
        echo '***********************************************************************************************'
        echo ' OpenMP is unavailable on this macOS system. LightGBM code will run single-threaded as a result.'
        echo ' To use all CPU cores for training jobs, you should install OpenMP by running'
        echo ''
        echo '     brew install libomp'
        echo '***********************************************************************************************'
    fi
fi

# substitute variables from this script into Makevars.in
AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIB)
AC_SUBST(LGB_CPPFLAGS)
AC_CONFIG_FILES([src/Makevars])

# write out Autoconf output
AC_OUTPUT
