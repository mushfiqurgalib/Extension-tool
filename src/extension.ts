import * as vscode from 'vscode';
import { execFile } from 'child_process';

export function activate(context: vscode.ExtensionContext) {

    let disposable = vscode.commands.registerCommand('extension.generateComment', () => {

        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const code = editor.document.getText(selection);
        const documentUri = editor.document.uri;
        const insertPosition = selection.start;

        if (!code) {
            vscode.window.showErrorMessage("No code selected!");
            return;
        }

        const pythonPath = "python";
        const scriptPath = vscode.Uri.joinPath(context.extensionUri, 'python', 'comment_generator.py').fsPath;

        const process = execFile(pythonPath, [scriptPath], (error, stdout, stderr) => {
            if (error) {
                vscode.window.showErrorMessage(`Error: ${stderr}`);
                return;
            }

            try {
                const result = JSON.parse(stdout);
                if (result.error) {
                    vscode.window.showErrorMessage(`Error: ${result.error}`);
                    return;
                }

                const comment = result.comment;
                const workspaceEdit = new vscode.WorkspaceEdit();
                workspaceEdit.insert(documentUri, insertPosition, comment + "\n");

                vscode.workspace.applyEdit(workspaceEdit).then(applied => {
                    if (!applied) {
                        vscode.window.showErrorMessage("Could not insert comment because the file changed or was closed.");
                    }
                });

            } catch {
                vscode.window.showErrorMessage("Failed to parse response");
            }
        });

        process.stdin?.write(code);
        process.stdin?.end();
    });

    context.subscriptions.push(disposable);
}

export function deactivate() { }
