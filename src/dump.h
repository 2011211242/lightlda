/*!
 * \file dump_binary.cpp
 * \brief Preprocessing tool for converting LibSVM data to LightLDA input binary format
 *  Usage: 
 *    dump_binary <libsvm_input> <word_dict_file_input> <binary_output_dir> <output_file_offset>
 */
#ifndef	_LIGHTLDA_DUMP_BINARY_
#define _LIGHTLDA_DUMP_BINARY_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
namespace multiverso { namespace lightlda
{
    /* 
     * Output file format:
     * 1, the first 8 byte indicates the number of docs in this block
     * 2, the 8 * (doc_num + 1) bytes indicate the offset of reach doc
     * an example
     * 3    // there are 3 docs in this block
     * 0    // the offset of the 1-st doc
     * 10   // the offset of the 2-nd doc, with this we know the length of the 1-st doc is 5 = 10/2
     * 16   // the offset of the 3-rd doc, with this we know the length of the 2-nd doc is 3 = (16-10)/2
     * 24   // with this, we know the length of the 3-rd doc is 4 = (24 - 16)/2
     * w11 t11 w12 t12 w13 t13 w14 t14 w15 t15  // the token-topic list of the 1-st doc
     * w21 t21 w22 t22 w23 t23                     // the token-topic list of the 2-nd doc
     * w31 t31 w32 t32 w33 t33 w34 t34             // the token-topic list of the 3-rd doc
         
    (1) open an utf-8 encoded file in binary mode,
    get its content line by line. Working around the CTRL-Z issue in Windows text file reading.
    (2) assuming each line ends with '\n'
    */

	struct Topic_token
	{
		int32_t topic_id;
		int32_t count;
	};

    class utf8_stream
    {
    public:
        utf8_stream();
        ~utf8_stream();

        bool open(const std::string& file_name);

        /*
        return true if successfully get a line (may be empty), false if not.
        It is user's task to verify whether a line is empty or not.
        */
        bool getline(std::string &line);
        int64_t count_line();
        bool close();
    private:
        bool block_is_empty();
        bool fill_block();
        std::ifstream stream_;
        std::string file_name_;
        const int32_t block_buf_size_ = 1024 * 1024;
        // const int32_t block_buf_size_ = 2;
        std::string block_buf_;
        std::string::size_type buf_idx_;
        std::string::size_type buf_end_;

        utf8_stream(const utf8_stream& other) = delete;
        utf8_stream& operator=(const utf8_stream& other) = delete;
    };

	class dump{
	public:
		dump(std::string input_dir);
		~dump();
        int32_t size_summary();
		void load_global_tf(std::string word_tf_file);
		void load_word_topic(std::string word_topic_file);
		void binary_dump(const std::vector<std::string>& word_list);
		//void binary_dump(std::string& libsvm_file_name);
		int32_t get_vocab_num() const{return vocab_num;}
		int32_t get_doc_buf_size() const{return doc_buf_size;}
		const int32_t* get_doc_buf()const{return doc_buf;}
		const int32_t* get_vocab_buf() const{return vocab_buf;} 
		const int32_t* get_local_tf_buf() const{return local_tf_buf;}
		const int32_t* get_global_tf_buf()const{return global_tf_buf;}
		void generate_files() const;
		const std::vector<int32_t>& get_local_words() const;
		const std::vector<Topic_token>& get_topics(int32_t word_id);
		const std::vector<int32_t>& get_summary()const;

	private:
		std::vector<int32_t> global_tf_map;
		std::unordered_map<std::string, int32_t> global_word_id_map;
		std::vector<int32_t> word_topic_map;
		std::vector<int32_t> local_words_;
		std::vector<std::vector<Topic_token>> word_topic_table;
		std::vector<int32_t> topic_summary;

		//doc dump_binary
		int32_t * doc_buf; 		
		//doc vocab id collection
		int32_t * vocab_buf;
		//local vacab term frequency
		int32_t * local_tf_buf;
		//global vacab term frequency
		int32_t * global_tf_buf;

		int32_t vocab_num;
		int32_t doc_buf_size;
		//const int32_t  kMaxDocLength;
		//const int32_t  kMaxDocBufLength;
	};
}
}

struct Token {
	int32_t word_id;
   	int32_t topic_id;
};

int Compare(const Token& token1, const Token& token2);
void split_string(std::string& line, char separator, std::vector<std::string>& output, bool trimEmpty = false);
std::vector<std::string> get_doc_tokens(std::string file_name);
std::vector<std::string> get_line_tokens(std::string &line);

#endif
