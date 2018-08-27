from bs4 import BeautifulSoup

class MessageThread:
    """This is a class to represent a Facebook message thread.

    Attributes:
        users (list of str): List of thread participants.
        messages (list of str): Ordered list of messages.
        senders (list of str): Ordered list of senders corresponding to 
            same index of self.messages.
        times (list of str): Time stamps for each message in self.messages.
        participants (set of str): Not really sure why I added this instead of users,
            maybe because self.users is non-unique so I will keep it.
        user_count (int): Total number of participants.
    """

    def __init__(self, users, messages, senders, times=None):
        """Initializes MessageThread object"""
        self.users = users
        self.messages = messages
        self.senders = senders
        self.times = times

        self.participants = set(self.senders)
        self.user_count = len(self.participants)


    def __getitem__(self, key):
        """Retrieve the i-th message in thread.

        Args:
            key (int): The i-th message in thread to be returned.

        Returns:
            str: Sender of i-th message.
            str: Content of i-th message.
        """
        return self.senders[key], self.messages[key]

    def __str__(self):
        """Sets formatting when printing MessageThread object.

        Example:
            'Kevin: hey have any dank memes for me today
            Yan: no
            Ryan: i'm going to sleep'
        """
        if self.times is None:
            return ''.join([sender +': ' + message + '\n' for sender, message in zip(self.senders, self.messages)])
        else:
            return ''.join(['[' + time + '] ' + sender +': ' + message + '\n' for time, sender, message in zip(self.times, self.senders, self.messages)])

    def crunch(self):
        """If there are multiple consecutive messages from the same sender, this function
        will join them into a single message and return an updated MessageThread object.
        """
        str_buff = []
        prev = None

        senders = []
        messages = []

        for message, sender in zip(self.messages, self.senders):
            if sender != prev:
                if prev is not None:
                    messages.append(' '.join(str_buff))
                    senders.append(prev)
                    str_buff = []
            str_buff.append(message)
            prev = sender

        messages.append(' '.join(str_buff))
        senders.append(prev)
        return MessageThread(self.users, messages, senders)

    def clean(self):
        """Removes empty or corrupted messages"""
        senders = []
        messages = []

        for message, sender in zip(self.messages, self.senders):
            if message:
                senders.append(sender)
                messages.append(message)

        return MessageThread(self.users, messages, senders)


class MessageHistory:
    """This class is a representation of all MessageThreads of a single user.

    Attributes:
        threads (list of MessageThread): All message threads for a user.
    """
    def __init__(self, path):
        """Parses html message file downloadable through facebook, 'messages.htm'
        and initializes MessageHistory object.

        Args:
            path (str): Path to messages.htm file from Facebook
        """
        with open(path) as fp:
            soup = BeautifulSoup(fp, 'html.parser')
        self.threads = self.parse(soup)

    def parse(self, soup):
        """Given BeautifulSoup object, extracts message history to
        create list of threads.

        Args:
            soup (BeatifulSoup): Raw html data from BeautifulSoup pkg.
        """
        threads_raw = soup.findAll('div', {'class': 'thread'})
        threads = []
        for thread in threads_raw:
            users = thread.find(text=True, recursive=False).split(', ')

            messages_raw = thread.findAll('p')
            senders_raw = thread.findAll('span', {'class': 'user'})
            times_raw = thread.findAll('span', {'class': 'meta'})

            messages = [messages_raw[i].text for i in reversed(range(len(messages_raw)))]
            senders = [senders_raw[i].text for i in reversed(range(len(senders_raw)))]
            times = [times_raw[i].text for i in reversed(range(len(times_raw)))]

            threads.append(MessageThread(users, messages, senders, times))
        return threads

    def get_thread_by_user(self, user, single=False):
        """Given a user, returns all threads associated with that user.

        Args:
            user (str): Name of user to find threads associated with.
            single (bool): If true, obtains only direct message (not group convo).

        Returns:
            list of int: Indices of threads containing specified user.
            list of MessageThread: List of MessageThread objects for specified user.
        """
        indices = []
        threads = []
        for idx, thread in enumerate(self.threads):
            if user in thread.users:
                if single:
                    if len(thread.users) == 2:
                        indices.append(idx)
                        threads.append(thread)
                else:
                    indices.append(idx)
                    threads.append(thread)

        return indices, threads

    def get_direct_message_threads(self):
        """Retrieves all MessageThread objects with only two participants (the user
        and one other individual), i.e. direct messages.
        """
        threads = []
        for thread in self.threads:
            if thread.user_count == 2:
                threads.append(thread)

        return threads
