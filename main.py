from __future__ import annotations
from dataclasses import dataclass
import enum


class NodeVisitor:
    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        f = getattr(self, method, self._generic_visit)
        return f(node)

    def _generic_visit(self, node):
        pass


@enum.unique
class TokenKind(enum.Enum):
    UNKNOWN = enum.auto()
    END_OF_FILE = enum.auto()

    LINE_COMMENT = enum.auto()
    WHITESPACE = enum.auto()
    IDENTIFIER = enum.auto()

    L_PAREN = enum.auto()
    R_PAREN = enum.auto()

    L_BRACE = enum.auto()
    R_BRACE = enum.auto()

    COLON = enum.auto()
    COMMA = enum.auto()

    ARROW = enum.auto()
    EQUAL = enum.auto()

    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()

    INT_LIT = enum.auto()

    KW_OPEN = enum.auto()
    KW_FUNC = enum.auto()
    KW_LET = enum.auto()
    KW_RETURN = enum.auto()


KEYWORD_MAP = {
    "open": TokenKind.KW_OPEN,
    "func": TokenKind.KW_FUNC,
    "let": TokenKind.KW_LET,
    "return": TokenKind.KW_RETURN,
}


@dataclass
class Token:
    kind: TokenKind
    text: str
    virtual: bool


def is_whitespace(c):
    return c == " " or c == "\n"


def is_ident_start(c):
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or (c == "_")


def is_ident_cont(c):
    return (
        (c >= "a" and c <= "z")
        or (c >= "A" and c <= "Z")
        or (c == "_")
        or (c >= "0" and c <= "9")
    )

def is_int_lit_start(c):
    return c >= "0" and c <= "9"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.index = 0
        self.start = 0

    def is_end(self):
        return self.index >= len(self.source)

    def peek(self):
        if self.is_end():
            return None
        return self.source[self.index]

    def peek2(self):
        if self.index + 1 >= len(self.source):
            return None
        return self.source[self.index + 1]

    def advance(self, n: int = 1):
        if self.is_end():
            return
        self.index += n

    def eat_while(self, f):
        while not self.is_end():
            c = self.peek()
            if not f(c):
                break
            self.advance()

    def make_token(self, kind: TokenKind):
        start = self.start
        end = self.index
        self.start = self.index
        return Token(kind, self.source[start:end], virtual=False)

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_end():
            raise StopIteration
        return self.next_token()

    def next_token(self):
        ch = self.peek()
        if ch is None:
            return self.make_token(TokenKind.END_OF_FILE)
        if is_whitespace(ch):
            return self.lex_whitespace()
        if is_ident_start(ch):
            return self.lex_identfier()
        if is_int_lit_start(ch):
            return self.lex_int_lit()
        match ch:
            case "(":
                self.advance()
                return self.make_token(TokenKind.L_PAREN)
            case ")":
                self.advance()
                return self.make_token(TokenKind.R_PAREN)
            case "{":
                self.advance()
                return self.make_token(TokenKind.L_BRACE)
            case "}":
                self.advance()
                return self.make_token(TokenKind.R_BRACE)
            case ":":
                self.advance()
                return self.make_token(TokenKind.COLON)
            case ",":
                self.advance()
                return self.make_token(TokenKind.COMMA)
            case "=":
                self.advance()
                return self.make_token(TokenKind.EQUAL)
            case "+":
                self.advance()
                return self.make_token(TokenKind.ADD)
            case "-":
                ch2 = self.peek2()
                match ch2:
                    case ">":
                        self.advance(2)
                        return self.make_token(TokenKind.ARROW)
                    case _:
                        self.advance()
                        return self.make_token(TokenKind.SUB)
            case "*":
                self.advance()
                return self.make_token(TokenKind.MUL)
            case "/":
                ch2 = self.peek2()
                if ch2 == "/":
                    return self.lex_line_comment()
                self.advance()
                return self.make_token(TokenKind.DIV)
        return self.lex_unknown()

    def lex_whitespace(self):
        self.eat_while(is_whitespace)
        return self.make_token(TokenKind.WHITESPACE)

    def lex_unknown(self):
        self.advance()
        return self.make_token(TokenKind.UNKNOWN)

    def lex_identfier(self):
        self.eat_while(is_ident_cont)
        token = self.make_token(TokenKind.IDENTIFIER)
        keyword = KEYWORD_MAP.get(token.text)
        if keyword is not None:
            token.kind = keyword
        return token

    def lex_line_comment(self):
        self.advance(2)
        self.eat_while(lambda c: not c == "\n")
        return self.make_token(TokenKind.LINE_COMMENT)

    # TODO: Different base.
    def lex_int_lit(self):
        self.eat_while(lambda c: c >= "0" and c <= "9")
        return self.make_token(TokenKind.INT_LIT)


def lex(source: str):
    lexer = Lexer(source)
    for tok in lexer:
        print(tok)


@enum.unique
class SyntaxKind(enum.Enum):
    UNKNOWN = enum.auto()

    SOURCE_FILE = enum.auto()

    OPEN_DECL = enum.auto()
    OPEN_PATH = enum.auto()

    FUNC_DECL = enum.auto()

    FUNC_SIG = enum.auto()
    FUNC_PARAM_LIST = enum.auto()
    FUNC_PARAM = enum.auto()
    FUNC_RET = enum.auto()

    LET_DECL = enum.auto()
    INIT_CLAUSE = enum.auto()

    BLOCK = enum.auto()

    TYPE_ANNOTATION = enum.auto()

    TYPE_EXPR = enum.auto()

    INT_LIT = enum.auto()


@dataclass
class SyntaxNode:
    kind: SyntaxKind
    children: list[SyntaxNode | Token]


class SyntaxBuilder:
    # The initial state is a source file node.
    def __init__(self):
        self.states = []
        self.current = (SyntaxKind.SOURCE_FILE, [])

    def start_node(self, kind: SyntaxKind):
        self.states.append(self.current)
        self.current = (kind, [])

    def end_node(self):
        kind, children = self.current
        node = SyntaxNode(kind=kind, children=children)
        self.current = self.states.pop()
        self.push_node(node)

    def push_node(self, node: SyntaxNode):
        _, children = self.current
        children.append(node)

    def push_token(self, token: Token):
        _, children = self.current
        children.append(token)

    def build(self):
        assert len(self.states) == 0
        kind, children = self.current
        return SyntaxNode(kind=kind, children=children)

def is_trivia_token(tok: Token):
    return tok.kind == TokenKind.WHITESPACE or tok.kind == TokenKind.LINE_COMMENT

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = lexer.next_token()
        self.builder = SyntaxBuilder()

    def advance(self):
        self.push_token(self.current_token)
        self.current_token = self.lexer.next_token()

    def is_end(self) -> bool:
        return self.current_token.kind == TokenKind.END_OF_FILE

    def start_node(self, kind: SyntaxKind):
        self.builder.start_node(kind)

    def end_node(self):
        self.builder.end_node()

    def push_token(self, kind: TokenKind):
        self.builder.push_token(kind)

    def parse(self) -> SyntaxNode:
        self.parse_source_file()
        return self.builder.build()

    def advance_while(self, f):
        while not self.is_end():
            tok = self.current_token
            if not f(tok):
                break
            self.advance()

    def consume(self, kind: TokenKind):
        if self.current_token.kind == kind:
            self.advance()
            return
        self.builder.push_token(Token(kind, text="", virtual=True))

    def consume_optional(self, kind: TokenKind):
        if not self.current_token.kind == kind:
            return
        self.advance()

    def skip_trivia(self):
        self.advance_while(is_trivia_token)

    def parse_unknown(self):
        self.start_node(SyntaxKind.UNKNOWN)
        self.advance()
        self.end_node()

    def parse_source_file(self):
        items = []
        while not self.is_end():
            self.skip_trivia()
            if self.current_token.kind == TokenKind.END_OF_FILE:
                break
            self.parse_source_file_items()

    def parse_source_file_items(self):
        match self.current_token.kind:
            case TokenKind.KW_OPEN:
                self.parse_open_decl()
            case TokenKind.KW_FUNC:
                self.parse_func_decl()
            case _:
                self.parse_unknown()

    def parse_open_decl(self):
        self.start_node(SyntaxKind.OPEN_DECL)
        self.consume(TokenKind.KW_OPEN)
        self.skip_trivia()
        # TODO
        self.parse_open_path()
        self.end_node()

    def parse_open_path(self):
        self.start_node(SyntaxKind.OPEN_PATH)
        self.consume(TokenKind.IDENTIFIER)
        self.end_node()

    def parse_func_decl(self):
        self.start_node(SyntaxKind.FUNC_DECL)
        self.consume(TokenKind.KW_FUNC)
        self.skip_trivia()
        # TODO
        self.consume(TokenKind.IDENTIFIER)
        self.skip_trivia()
        self.parse_func_sig()
        self.skip_trivia()
        if self.current_token.kind == TokenKind.L_BRACE:
            self.parse_block()
        self.end_node()

    def parse_func_sig(self):
        self.start_node(SyntaxKind.FUNC_SIG)
        self.consume(TokenKind.L_PAREN)
        self.skip_trivia()
        self.parse_func_param_list()
        self.skip_trivia()
        self.consume(TokenKind.R_PAREN)
        self.skip_trivia()
        if self.current_token.kind == TokenKind.ARROW:
            self.parse_func_ret()
        self.end_node()

    def parse_func_param_list(self):
        self.start_node(SyntaxKind.FUNC_PARAM_LIST)
        while not self.is_end():
            self.skip_trivia()
            if self.current_token.kind == TokenKind.R_PAREN:
                break
            self.parse_func_param()
            self.consume_optional(TokenKind.COMMA)
            self.skip_trivia()
        self.end_node()

    def parse_func_param(self):
        self.start_node(SyntaxKind.FUNC_PARAM)
        self.consume(TokenKind.IDENTIFIER)
        self.skip_trivia()
        if self.current_token.kind == TokenKind.COLON:
            self.parse_type_annotation()
        self.end_node()

    def parse_func_ret(self):
        self.start_node(SyntaxKind.FUNC_RET)
        self.consume(TokenKind.ARROW)
        self.skip_trivia()
        self.parse_type_expr()
        self.end_node()

    def parse_block(self):
        self.start_node(SyntaxKind.BLOCK)
        self.consume(TokenKind.L_BRACE)
        while not self.is_end():
            self.skip_trivia()
            if self.current_token.kind == TokenKind.R_BRACE:
                break
            self.parse_block_item()
        self.consume(TokenKind.R_BRACE)
        self.end_node()

    def parse_block_item(self):
        match self.current_token.kind:
            case TokenKind.KW_LET:
                self.parse_let_decl()
            case _:
                self.parse_unknown()

    def parse_let_decl(self):
        self.start_node(SyntaxKind.LET_DECL)
        self.consume(TokenKind.KW_LET)
        self.skip_trivia()
        self.consume(TokenKind.IDENTIFIER)
        self.skip_trivia()
        if self.current_token.kind == TokenKind.COLON:
            self.parse_type_annotation()
        self.skip_trivia()
        if self.current_token.kind == TokenKind.EQUAL:
            self.parse_init_clause()
        self.end_node()

    def parse_init_clause(self):
        self.start_node(SyntaxKind.INIT_CLAUSE)
        self.consume(TokenKind.EQUAL)
        self.skip_trivia()
        self.parse_expr()
        self.end_node()

    def parse_type_expr(self):
        self.start_node(SyntaxKind.TYPE_EXPR)
        self.consume(TokenKind.IDENTIFIER)
        self.end_node()

    def parse_type_annotation(self):
        self.start_node(SyntaxKind.TYPE_ANNOTATION)
        self.consume(TokenKind.COLON)
        self.skip_trivia()
        self.parse_type_expr()
        self.end_node()

    def parse_expr(self):
        self.expr_leading()

    def expr_leading(self):
        match self.current_token.kind:
            case TokenKind.INT_LIT:
                self.begin_node(SyntaxKind.INT_LIT)
                self.advance()
                self.end_node()
            case _:
                pass


def parse(source: str):
    parser = Parser(Lexer(source))
    return parser.parse()


# node:
def pp(node: SyntaxNode | Token, indent=0):
    prefix = " " * indent
    info = f"{prefix}{node.kind}"
    if isinstance(node, Token):
        if node.virtual:
            info += "  <-- virtual"
        elif not node.kind == TokenKind.WHITESPACE:
            info += f"  {node.text}"
    print(info)
    if isinstance(node, SyntaxNode):
        for child in node.children:
            pp(child, indent=indent + 4)


pp(
    parse("""
open std
func foo(x: int, y: bool, z) -> int {
    let w: f32 = 012
}
""")
)


def main():
    pass


if __name__ == "__main__":
    main()
