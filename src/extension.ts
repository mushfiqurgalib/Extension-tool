import * as vscode from 'vscode';
import { execFile } from 'child_process';

function toDocumentPosition(selectionStart: vscode.Position, relativeLine: number, relativeCharacter: number): vscode.Position {
    if (relativeLine === 0) {
        return new vscode.Position(selectionStart.line, selectionStart.character + relativeCharacter);
    }

    return new vscode.Position(selectionStart.line + relativeLine, relativeCharacter);
}

export function activate(context: vscode.ExtensionContext) {
    const outputChannel = vscode.window.createOutputChannel('AI Comment Generator');

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
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine('Running comment generator...');

        const process = execFile(pythonPath, [scriptPath], (error, stdout, stderr) => {
            if (stderr) {
                outputChannel.append(stderr);
            }

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

                if (!result.edit && result.comment) {
                    const workspaceEdit = new vscode.WorkspaceEdit();
                    workspaceEdit.insert(documentUri, selection.start, `${result.comment}\n`);

                    vscode.workspace.applyEdit(workspaceEdit).then(applied => {
                        if (!applied) {
                            vscode.window.showErrorMessage("Could not insert comment because the file changed or was closed.");
                        }
                    });
                    return;
                }

                if (!result.edit || result.edit.type === 'none') {
                    vscode.window.showInformationMessage(result.message ?? 'Documentation is already adequate.');
                    return;
                }

                const workspaceEdit = new vscode.WorkspaceEdit();
                const start = toDocumentPosition(selection.start, result.edit.startLine, result.edit.startCharacter);
                const end = toDocumentPosition(selection.start, result.edit.endLine, result.edit.endCharacter);
                const replacementRange = new vscode.Range(start, end);

                if (result.edit.type === 'replace') {
                    workspaceEdit.replace(documentUri, replacementRange, result.edit.text);
                } else {
                    workspaceEdit.insert(documentUri, start, result.edit.text);
                }

                vscode.workspace.applyEdit(workspaceEdit).then(applied => {
                    if (!applied) {
                        vscode.window.showErrorMessage("Could not insert comment because the file changed or was closed.");
                        return;
                    }

                    if (result.message) {
                        vscode.window.showInformationMessage(result.message);
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
    context.subscriptions.push(outputChannel);
}

export function deactivate() { }
