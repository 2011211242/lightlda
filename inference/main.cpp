#include "infer.h"

void Run(int argc, char** argv)
{
    using namespace multiverso::lightlda;
    Infer::Init();
    std::string input_file("/search/odin/yanxianlong/lightLDA/lightlda-master/data/out.txt");
    multiverso::StopWatch watch; 
    watch.Start();
    double elapsedSeconds = 0.0000f;
    utf8_stream doc_stream;
    doc_stream.open(input_file);

    std::string line;
    while(doc_stream.getline(line))
    {
        std::vector<std::string> tokens_input;
        tokens_input = get_line_tokens(line);
        watch.Restart();
        std::vector<std::pair<int32_t, int32_t>> res;
        res = Infer::predict(tokens_input);
        elapsedSeconds += watch.ElapsedSeconds();
    }
    Infer::Clear();
    multiverso::Log::Info("inferers Time used: %.2f s \n", elapsedSeconds);
}

int main(int argc, char** argv)
{
    Run(argc, argv);
    return 0;
}
