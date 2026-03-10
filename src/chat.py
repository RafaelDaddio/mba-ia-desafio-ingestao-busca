from search import search_prompt
from langchain_core.runnables import RunnableLambda

def main():
    search = RunnableLambda(search_prompt)

    if not search:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    while True:
        user_input = input("\nFaça uma pergunta (ou digite 'sair' para encerrar): ")
        if user_input.lower() == "sair":
            print("Encerrando o chat. Até mais!")
            break
        
        response = search.invoke(input=user_input)
        print(f"\nResposta:\n{response}")

if __name__ == "__main__":
    main()