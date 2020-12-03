Sort::Naturally is required by `mteval-v14.pl` and `wrap-xml.perl` scripts and the two scripts have been modified with \
the following lines to be accessed locally when called through subprocess 
(if the package is not accessible through the local installation of perl):
```
use FindBin 1.51 qw( $RealBin );
use lib $RealBin;
```
