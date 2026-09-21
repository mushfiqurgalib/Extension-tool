import * as vscode from 'vscode';
import { execFile } from 'child_process';

function toDocumentPosition(selectionStart: vscode.Position, relativeLine: number, relativeCharacter: number): vscode.Position {
    if (relativeLine === 0) {
        return new vscode.Position(selectionStart.line, selectionStart.character + relativeCharacter);
    }

    return new vscode.Position(selectionStart.line + relativeLine, relativeCharacter);
}

function indentPlainComment(comment: string, indentLevel: number): string {
    const indent = ' '.repeat(indentLevel);
    return comment
        .split('\n')
        .map(line => line.length > 0 ? `${indent}${line}` : indent)
        .join('\n');
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
        const scriptPath = vscode.Uri.joinPath(context.extensionUri, 'python', 'comment_generator_ollama.py').fsPath;
        outputChannel.clear();
        outputChannel.show(true);
        outputChannel.appendLine('Running comment generator...');

        const process = execFile(pythonPath, [scriptPath], (error, stdout, stderr) => {
            if (stderr) {
                outputChannel.append(stderr);
            }

            if (error) {
                const errorMsg = stderr.trim().startsWith('Error:') ? stderr.trim() : `Error: ${stderr.trim()}`;
                vscode.window.showErrorMessage(errorMsg);
                return;
            }

            try {
                const result = JSON.parse(stdout);
                if (result.error) {
                    const errorMsg = result.error.trim().startsWith('Error:') ? result.error.trim() : `Error: ${result.error.trim()}`;
                    vscode.window.showErrorMessage(errorMsg);
                    return;
                }

                if (!result.edit && result.comment) {
                    const workspaceEdit = new vscode.WorkspaceEdit();
                    const insertionStart = new vscode.Position(selection.start.line, 0);
                    const formattedComment = indentPlainComment(result.comment, selection.start.character);
                    workspaceEdit.insert(documentUri, insertionStart, `${formattedComment}\n`);

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
