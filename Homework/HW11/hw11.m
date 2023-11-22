%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g2 = @(t) t.^(2 - 1) .* exp((-1) .* t);
g4 = @(t) t.^(4 - 1) .* exp((-1) .* t);
g6 = @(t) t.^(6 - 1) .* exp((-1) .* t);
g8 = @(t) t.^(8 - 1) .* exp((-1) .* t);
g10 = @(t) t.^(10 - 1) .* exp((-1) .* t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Problem 3b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("Problem 3b)")
disp("")

[q2, evals2] = quad(g2, 0, 100);
[q4, evals4] = quad(g4, 0, 100);
[q6, evals6] = quad(g6, 0, 100);
[q8, evals8] = quad(g8, 0, 100);
[q10, evals10] = quad(g10, 0, 100);

disp("x = 2")
disp("MATLAB quad: ")
disp(q2)
disp("Number of function calls: ")
disp(evals2)
disp("")

disp("x = 4")
disp("MATLAB quad: ")
disp(q4)
disp("Number of function calls: ")
disp(evals4)
disp("")

disp("x = 6")
disp("MATLAB quad: ")
disp(q6)
disp("Number of function calls: ")
disp(evals6)
disp("")

disp("x = 8")
disp("MATLAB quad: ")
disp(q8)
disp("Number of function calls: ")
disp(evals8)
disp("")

disp("x = 10")
disp("MATLAB quad: ")
disp(q10)
disp("Number of function calls: ")
disp(evals10)
disp("")
