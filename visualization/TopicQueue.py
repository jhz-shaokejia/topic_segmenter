
from Queue import PriorityQueue
from collections import namedtuple



Entry = namedtuple('Entry', ['priority', 'wordset', 'id'])


class TopicCover():
    """docstring for TopicQueue"""

    def __init__(self, messages):
        """ Generates a TopicCover class for a set of messages
        Inputs:
            messages: the message dictionary: { id: text, ... }
        """
        self.pq = PriorityQueue()
        self.cover = []
        self.words_in_cover = set()

        # add message dictionary and process all messages (add to priority queue)
        self.message_corpus = messages
        # TODO: process messages prior to ingestion
        for msg_id in self.message_corpus.iterkeys():
            self.add_entry(msg_id)


    def add_entry(self, message_id):
        """ Add a message to the topic queue """
        message_words = set(self.message_corpus[message_id].split())
        entry = Entry(priority=-len(message_words), wordset=message_words, id=message_id)
        self.pq.put( entry )


    def update_entry(self, entry):
        """ Updates the priority and wodset of an entry based on the actual cover """
        new_words = entry.wordset.difference(self.words_in_cover)
        return Entry(priority=-len(new_words), wordset=new_words, id=entry.id)


    def stop_updating(self, entry):
        """ Checks if the priority queue has entries with potentially lower priority """
        try:
            return self.pq.queue[0].priority >= entry.priority
        except IndexError:
            # In case the priority queue is empty return True
            return True


    def increment_cover(self):
        """ Update cover with one more message """
        while True:
            # get the best entry and update it to ge the new priority and wordset
            popped_entry = self.pq.get()
            new_entry = self.update_entry(popped_entry)
            if self.stop_updating(new_entry):
                break
            else:
                # if it is not the best entry, put back in the priority queue
                self.pq.put( new_entry )

        # add to cover, and record words
        self.cover.append( self.message_corpus[new_entry.id] )
        self.words_in_cover.update( new_entry.wordset )


    def get_cover(self, min_messages):
        """ Returns the N-message cover, computed with a greedy algorithm """
        if len(self.message_corpus) < min_messages:
            raise ValueError('Cannot obtain a cover bigger than the message corpus!')

        if len(self.cover) < min_messages:
            while len(self.cover) <= min_messages:
                self.increment_cover()
        return self.cover[:min_messages]

