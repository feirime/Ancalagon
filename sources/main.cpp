#include "run.h"

int main(int argc, char* argv[])
{
    Run *run = new Run;
    run->run(argc, argv);
    return 0;
}
