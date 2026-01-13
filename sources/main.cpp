#include "run.h"

int main(int argc, char* argv[])
{
    auto *run = new RunMultiRadiuses;
    run->run(argc, argv);
    return 0;
}
