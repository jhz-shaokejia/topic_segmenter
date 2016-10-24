import sys
import os
import math
import pandas as pd

from text.Message import Message
from grammar.MessageTokenizer import MessageTokenizer
from segmenter.ConversationSegmenter import ConversationSegmenter
from text.JSONParser import JSONParser

class SegmenterRunner:
    def __init__(self, json_file_name, output_folder=None):
        self.json_file_name = json_file_name
        self.output_folder = output_folder
        self.topics_table = None


    def run(self):
        parser = JSONParser(self.json_file_name)
        self.messages = parser.getMessages()
        self.tokenizer = MessageTokenizer()
        windowSize = 3
        cosineSimilarityThreshold = 0.8
        segmenter = ConversationSegmenter(
            self.messages, windowSize, cosineSimilarityThreshold, self.tokenizer)
        self.topics = segmenter.segment()

        self.build_table()

        if self.output_folder is not None:
            self.report_table()
        else:
            self.report()


    def build_table(self):
        """ Builds a table for each of topics """
        TOTAL_TOPICS = len(self.topics)

        for i, topic in enumerate(self.topics):

            ## TODO: get-topic-name
            topic_name = 'topic-{num:0{l}}'.format(num=i, l=int(math.floor(math.log10(TOTAL_TOPICS))) + 1)

            _messages = topic.getMessages()
            _reasons = topic.getReasons()

            topic_table = pd.DataFrame({'ID': map(lambda m: m.getID(), _messages),
                                        'text': map(lambda m: m.getText(), _messages),
                                        'reason': _reasons })
            topic_table['topic'] = topic_name

            # append to list of topic-tables
            if self.topics_table is not None:
                # merge topic into table
                self.topics_table = self.topics_table.append(topic_table, ignore_index=True)
            else:
                self.topics_table = topic_table


    def report_table(self):
        # Check existence of the output folder and create if necessary
        if not os.path.exists(self.output_folder):
            print(' - The specified folder was not found... folder will be created: \033[36m{}\033[0m')
            os.makedirs(self.output_folder)

        # Parse output path/file name and save table to topics_CHANNEL.csv
        filename = self.json_file_name.split('/')[-1].replace('.json', '')
        folderpath = self.output_folder[:-1] if self.output_folder.endswith('/') else self.output_folder
        out_path = '{path}/topics_{name}.csv'.format(path=folderpath, name=filename)
        self.topics_table.to_csv(out_path, encoding='utf-8')

        # Report output table
        print('  --> Output Topic table: \033[32m {} \033[0m'.format(out_path))


    def report(self):
        idGroups = []
        print("============================= detailed ========================")
        for topic in self.topics:
            print("== Topic ==")
            idGroup = []
            for (message, reason) in zip(topic.getMessages(), topic.getReasons()):
                idGroup.append(message.getID())
                print("\n\t------ id: \t" + str(message.getID()) + "\t" + reason)
                print("" + message.getText())
            print("\n")
            idGroups.append(idGroup)

        print("===============================")

        print("============================= short ========================")
        for topic in self.topics:
            print("== Topic ==")
            for message in topic.getMessages():
                print(str(message.getID()) + ":\t" + message.getText())
            print("\n")

        print(idGroups)



def main(json_input, output_folder=None):
    SegmenterRunner(json_input, output_folder).run()


if __name__ == '__main__':
    main(*sys.argv[1:])  # optionally might include a output_folder specification
