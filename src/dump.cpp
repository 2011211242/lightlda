#include "dump.h"
#include "common.h"
#include <multiverso/stop_watch.h>

using namespace multiverso::lightlda;

utf8_stream::utf8_stream()
{
    block_buf_.resize(block_buf_size_);
}

utf8_stream::~utf8_stream()
{
}

bool utf8_stream::open(const std::string& file_name)
{
    stream_.open(file_name, std::ios::in | std::ios::binary);
    buf_idx_ = 0;
    buf_end_ = 0;
    return stream_.good();
}

bool utf8_stream::getline(std::string& line)
{
    line = "";
    while (true)
    {
        if (block_is_empty())
        {
            // if the block_buf_ is empty, fill the block_buf_
            if (!fill_block())
            {
                // if fail to fill the block_buf_, that means we reach the end of file
                if (!line.empty())
                    std::cout << "Invalid format, according to our assumption: "
                   "each line has an \\n. However, we reach here with an non-empty line but not find an \\n";
                return false;
            }
        }
        // the block is not empty now

        std::string::size_type end_pos = block_buf_.find("\n", buf_idx_);
        if (end_pos != std::string::npos)
        {
            // successfully find a new line
            line += block_buf_.substr(buf_idx_, end_pos - buf_idx_);
            buf_idx_ = end_pos + 1;
            return true;
        }
        else
        {
            // do not find an \n untile the end of block_buf_
            line += block_buf_.substr(buf_idx_, buf_end_ - buf_idx_);
            buf_idx_ = buf_end_;
        }
    }
    return false;
}

int64_t utf8_stream::count_line()
{
    char* buffer = &block_buf_[0];

    int64_t line_num = 0;
    while (true)
    {
        stream_.read(buffer, block_buf_size_);
        int32_t end_pos = static_cast<int32_t>(stream_.gcount());
        if (end_pos == 0)
        {
            break;
        }
        line_num += std::count(buffer, buffer + end_pos, '\n');
    }
    return line_num;
}

bool utf8_stream::block_is_empty()
{
    return buf_idx_ == buf_end_;
}

bool utf8_stream::fill_block()
{
    char* buffer = &block_buf_[0];
    stream_.read(buffer, block_buf_size_);
    buf_idx_ = 0;
    buf_end_ = static_cast<std::string::size_type>(stream_.gcount());
    return buf_end_ != 0;
}

bool utf8_stream::close()
{
	stream_.close();
   	return true;
}

dump::dump(std::string input_dir)
{
	doc_buf = new int32_t[kMaxDocLength * 2 + 1];
	vocab_buf = new int32_t[kMaxDocLength];
	local_tf_buf = new int32_t[kMaxDocLength];
	global_tf_buf = new int32_t[kMaxDocLength];


	doc_buf_size = 0;
	vocab_num = 0;

	//load word_id_dict file
	std::string global_tf_file = input_dir + std::string("/word_id.dict");
	load_global_tf(global_tf_file);
	std::cout << "load_global_tf end!" << std::endl;

	//load word_topic file
	std::string word_topic_file = input_dir + std::string("/server_0_table_0.model");
	load_word_topic(word_topic_file);
	std::cout << "load_word_topic end!" << std::endl;
}

dump::~dump(){
	delete [] doc_buf;
	delete [] vocab_buf;
	delete [] local_tf_buf;
	delete [] global_tf_buf;
}

void dump::load_global_tf(std::string word_tf_file)
{
    global_tf_map.resize(Config::num_vocabs, 0);
    lightlda::utf8_stream stream;
    if (!stream.open(word_tf_file))
    {
        std::cout << "Fails to open file: " << word_tf_file << std::endl;
        exit(1);
    }
    std::string line;
    while (stream.getline(line))
    {
        std::vector<std::string> output;
        split_string(line, '\t', output);
        if (output.size() != 3)
        {
            std::cout << "Invalid line: " << line << std::endl;
            exit(1);
        }
        int32_t word_id = std::stoi(output[0]);
		std::string word = output[1];
        int32_t tf = std::stoi(output[2]);
        auto it = global_tf_map[word_id];
        if (it != 0)
        {
            std::cout << "Duplicate words detected: " << line << std::endl;
            exit(1);
        }

		auto it_ = global_word_id_map.find(word);
		if(it_ != global_word_id_map.end())
		{
			std::cout << "Duplicate words detected: " << line << std::endl;
            exit(1);
		}
		
        global_tf_map[word_id]=  tf;
		global_word_id_map.insert(std::make_pair(word, word_id));
    }
    stream.close();
}

void dump::load_word_topic(std::string word_topic_file)
{
	topic_summary.resize(1000, 0);
    word_topic_map.resize(Config::num_vocabs, 0);
    word_topic_table.resize(Config::num_vocabs);

	lightlda::utf8_stream stream;
    if (!stream.open(word_topic_file))
    {
        std::cout << "Fails to open file: " << word_topic_file << std::endl;
        exit(1);
    }

    std::string line;
    while (stream.getline(line))
    {
        std::vector<std::string> output;
        split_string(line, ' ', output);

        int32_t word_id = std::stoi(output[0]);

		std::vector<Topic_token> topic_tokens;
		int32_t topic_id = 0;
		int32_t topic_freq = 0;

        if(output.size() < 2)
        {
            std::cout << "this word has no topic related or format error!" << std::endl;
            exit(0);
        }

		for(int32_t i = 1; i < output.size(); i++)
		{
			std::vector<std::string> t_f;
			split_string(output[i], ':', t_f);

            if(t_f.size() < 2)
            {
                std::cout << "format error!" << std::endl;
                exit(0);
            }
   
			int32_t topic = std::stoi(t_f[0]);
			int32_t freq = std::stoi(t_f[1]);

			//caculate the topic has been show up in total
			topic_summary[topic] += freq;
			topic_tokens.push_back({topic, freq});

			if(freq > topic_freq)
			{
				topic_freq = freq;
				topic_id = topic;
			}
		}

        if (word_topic_map[word_id] > 0)
        {
            std::cout << "Duplicate words detected: " << line << std::endl;
            exit(1);
        }
		word_topic_map[word_id] = topic_id;
		word_topic_table[word_id] =  topic_tokens;	
    }
    stream.close();
}


const std::vector<Topic_token>& dump::get_topics(int32_t word_id)
{
	return word_topic_table[word_id];
}



const std::vector<int32_t>& dump::get_summary()const
{
	return topic_summary;
}


void dump::binary_dump(const std::vector<std::string>& word_list)
{

    int doc_token_count = 0;
	//std::unordered_map<int32_t,int32_t> local_tf_map;
	std::vector<Token> doc_tokens;

	for(auto& word : word_list)
	{
		if(global_word_id_map.find(word) != global_word_id_map.end()){
			int32_t word_id = global_word_id_map[word];
			int32_t topic_id = word_topic_map[word_id];
			doc_tokens.push_back({ word_id, 0});

		    ++doc_token_count;
            if (doc_token_count >= kMaxDocLength) break;
		}
	}

    // The input data may be already sorted
    std::sort(doc_tokens.begin(), doc_tokens.end(), Compare);

	int32_t doc_buf_idx = 0;
    doc_buf[doc_buf_idx++] = 0; // cursor

    vocab_num = -1;
    int32_t pre = -1;
	local_words_.resize(0);
    for (auto& token : doc_tokens)
    {
        //std::cout<<token.word_id<<std::endl;
        doc_buf[doc_buf_idx++] = token.word_id;
        doc_buf[doc_buf_idx++] = token.topic_id;
        if(pre != token.word_id)
        {
            vocab_num ++;
			local_words_.push_back(token.word_id);
            vocab_buf[vocab_num] = token.word_id;
			local_tf_buf[vocab_num] = 0;			
            global_tf_buf[vocab_num] = global_tf_map[token.word_id];

            pre = token.word_id;
        }
        local_tf_buf[vocab_num] ++;
    }	
    doc_buf_size = doc_buf_idx;
    vocab_num++;
}

void dump::generate_files() const
{
	std::string vocab_name("vocab.0");
	std::string block_name("block.0");

	std::ofstream vocab_file(vocab_name, std::ios::out | std::ios::binary);
	std::ofstream block_file(block_name, std::ios::out | std::ios::binary);	

	int32_t vocab_size = vocab_num;
	int64_t doc_num = 1;
	int64_t * offset_buf = new int64_t[doc_num + 1];
	offset_buf[0] = 0;
	offset_buf[1] = doc_buf_size;
	block_file.write(reinterpret_cast<char*>(&doc_num), sizeof(int64_t));
	block_file.write(reinterpret_cast<char*>(offset_buf), sizeof(int64_t) * (doc_num + 1));
	block_file.write(reinterpret_cast<char*>(doc_buf), sizeof(int32_t) * doc_buf_size);
	
	vocab_file.write(reinterpret_cast<char*>(&vocab_size), sizeof(int32_t));
	vocab_file.write(reinterpret_cast<char*>(vocab_buf), sizeof(int32_t) * vocab_size);
	vocab_file.write(reinterpret_cast<char*>(global_tf_buf), sizeof(int32_t) * vocab_size);
	vocab_file.write(reinterpret_cast<char*>(local_tf_buf), sizeof(int32_t) * vocab_size);

	vocab_file.close();
	block_file.close();
}

const std::vector<int32_t>& dump::get_local_words()const
{
	return local_words_;
}

/*
void dump::binary_dump(std::string& libsvm_file_name)
{
	lightlda::utf8_stream libsvm_file;

	if (!libsvm_file.open(libsvm_file_name))
    {
        std::cout << "Fails to open file: " << libsvm_file_name << std::endl;
        exit(1);
    }
	char *endptr = nullptr;
	std::string str_line;

	if (!libsvm_file.getline(str_line) || str_line.empty())
   	{
            std::cout << "Fails to get line" << std::endl;
            exit(1);
   	}
    str_line += '\n';

    std::vector<std::string> output;
    split_string(str_line, '\t', output);

    if (output.size() != 2)
    {
        std::cout << "Invalid format, not key space val: " << std::endl << str_line << std::endl;
        exit(1);
    }

    int doc_token_count = 0;

    char *ptr = &(output[1][0]);
	const int kBASE = 10;

	std::unordered_map<int32_t,int32_t> local_tf_map;
	std::vector<Token> doc_tokens;

    while (*ptr != '\n')
    {
        if (doc_token_count >= kMaxDocLength) break;
        // read a word_id:count pair
        int32_t word_id = strtol(ptr, &endptr, kBASE);
        ptr = endptr;
        if (':' != *ptr)
        {
            std::cout << "Invalid input" << str_line << std::endl;
            exit(1);
        }
        int32_t count = strtol(++ptr, &endptr, kBASE);

        ptr = endptr;
        for (int k = 0; k < count; ++k)
        {
			if(global_tf_map.find(word_id) != global_tf_map.end())
            {
				doc_tokens.push_back({ word_id, 0 });
            	if (local_tf_map.find(word_id) == local_tf_map.end())
				{
					local_tf_map.insert(std::make_pair(word_id, 1));
				}
				else
				{
					local_tf_map[word_id]++;
				}

				++doc_token_count;
            	if (doc_token_count >= kMaxDocLength) break;
		    }
 		}
        while (*ptr == ' ' || *ptr == '\r') ++ptr;
    }
    // The input data may be already sorted
    std::sort(doc_tokens.begin(), doc_tokens.end(), Compare);

	int32_t doc_buf_idx = 0;
    doc_buf[doc_buf_idx++] = 0; // cursor
    vocab_num = 0;
    int32_t pre = -1;
    for (auto& token : doc_tokens)
    {
        doc_buf[doc_buf_idx++] = token.word_id;
        doc_buf[doc_buf_idx++] = token.topic_id;
        if(pre != token.word_id)
        {
            vocab_buf[vocab_num] = token.word_id;
			local_tf_buf[vocab_num] = local_tf_map[token.word_id];
			global_tf_buf[vocab_num] = global_tf_map[token.word_id];
            pre = token.word_id;
            vocab_num ++;
        }
    }	
	doc_buf_size = doc_buf_idx;

	vocab_num = 0;
	for(int32_t i = 0; i < word_num; i++){
		if(local_tf_map[i] > 0){
			vocab_buf[vocab_num] = i;
			local_tf_buf[vocab_num] = local_tf_map[i];
			global_tf_buf[vocab_num] = global_tf_map[i];
			vocab_num++;
		}
	}	
}
*/


int Compare(const Token& token1, const Token& token2) {
    return token1.word_id < token2.word_id;
}

void split_string(std::string& line, char separator, std::vector<std::string>& output, bool trimEmpty)
{
    output.clear();

    if (line.empty())
    {
        return;
    }

    // trip whitespace, \r
    while (!line.empty())
    {
        int32_t last = line.length() - 1;
        if (line[last] == ' ' || line[last] == '\r')
        {
            line.erase(last, 1);
        }
        else
        {
            break;
        }
    }

    std::string::size_type pos;
    std::string::size_type lastPos = 0;

    using value_type = std::vector<std::string>::value_type;
    using size_type = std::vector<std::string>::size_type;

    while (true)
    {
        pos = line.find_first_of(separator, lastPos);
        if (pos == std::string::npos)
        {
            pos = line.length();

            if (pos != lastPos || !trimEmpty)
                output.push_back(value_type(line.data() + lastPos,
                (size_type)pos - lastPos));

            break;
        }
        else
        {
            if (pos != lastPos || !trimEmpty)
                output.push_back(value_type(line.data() + lastPos,
                (size_type)pos - lastPos));
        }

        lastPos = pos + 1;
    }
    return;
}

std::vector<std::string> get_doc_tokens(std::string file_name)
{
	utf8_stream doc_stream;
	doc_stream.open(file_name);

	std::string line;
	std::vector<std::string> token_input;
	while(doc_stream.getline(line))
    {
		std::vector<std::string> output;
		split_string(line, ' ', output);
		token_input.insert(std::end(token_input), std::begin(output), std::end(output));
	}
	return token_input;
}

std::vector<std::string> get_line_tokens(std::string &line)
{
    std::vector<std::string> tokens;
    split_string(line, ' ', tokens);
    return tokens;
}
