from transformers import AutoTokenizer

class ModifiedMambaTokenizerFactory:
    def __init__(self, student_tokenizer: AutoTokenizer, teacher_tokenizer: AutoTokenizer):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        teacher_vocab = set(teacher_tokenizer.get_vocab())
        student_vocab = set(student_tokenizer.get_vocab())
        student_tokenizer.add_tokens(list(teacher_vocab - student_vocab)[:len(teacher_vocab) - len(student_vocab)]) #
    

    def get_modified_tokenizer(self):
        return self.student_tokenizer