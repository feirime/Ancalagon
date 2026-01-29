#include "run.h"

int main(int argc, char* argv[])
{
    auto *run = new Run;
    run->run(argc, argv);
    return 0;
}
