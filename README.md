# SFUTranslate
This is an academic machine translation toolkit, in which the main focus has been towards readability and changeability.
We also have tried to make the algorithms as fast as possible, but please let us know if you have any suggestions or
concerns regarding the toolkit. To get familiar with what you can do and how you can do it please read through this document.

To run the code you will need python 3.5+.
 
# Resources
All the necessary resources for the project are supposed to be loaded from the `SFUTranslate/resources` directory.
Please refrain from putting the resources anywhere else if you are planing to do a pull request.
 
# Source Code
All the source code files are placed under `SFUTranslate/src` directory.
If you are using an IDE to debug or run the code, don't forget to mark this directory as your sources directory.
Otherwise you will need to have the address of `SFUTranslate/src` directory in your `$PATH` environment variable to be able to run the code.
Another way of running the code would be to run `cd /path/to/SFUTranslate/src && python -m package.subpackage.source_code_name <arguments>`.