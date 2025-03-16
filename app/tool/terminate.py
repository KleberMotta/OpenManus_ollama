from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """Termine a interação quando a solicitação for atendida OU quando você não puder prosseguir com a tarefa.
Use esta ferramenta para finalizar o trabalho quando todas as tarefas estiverem concluídas.
Inclua uma mensagem para o usuário usando o parâmetro 'message'."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "O status da interação.",
                "enum": ["success", "failure", "completed"],
                "default": "completed"
            },
            "message": {
                "type": "string",
                "description": "Mensagem que será mostrada ao usuário como resposta final.",
                "default": ""
            }
        },
        "required": ["message"]
    }

    async def execute(self, status: str = "completed", message: str = "", **kwargs) -> str:
        """Finaliza a execução atual e retorna uma mensagem para o usuário"""
        if message:
            return message
        else:
            return "Tarefa concluída."
