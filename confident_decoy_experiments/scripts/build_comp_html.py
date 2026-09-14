import json, html, sys
M = sys.argv[1]
CFG = {"fnat": ("fnat", "fnat (fraction of native contacts recovered)",
                "the shaded band is fnat &ge; 0.30, i.e. the pose recovered at least 30% of the true contacts",
                "fnat &ge; 0.30"),
       "irmsd": ("iRMSD", "interface RMSD (&Aring;)",
                 "the pink band is iRMSD &le; 2 &Aring;, the CAPRI medium-accuracy interface criterion",
                 "iRMSD &le; 2 &Aring;"),
       "lrmsd": ("LRMSD", "ligand RMSD (&Aring;)",
                 "the pink band is LRMSD &le; 5 &Aring;, the CAPRI medium-accuracy placement criterion",
                 "LRMSD &le; 5 &Aring;")}[M]
src = open("scripts/build_h3_slideshow.py").read()
s = src.replace('artifacts/h3_panels/summary.json', f'artifacts/{M}_panels/summary.json')
s = s.replace('artifacts/h3_panels/meta.json', f'artifacts/{M}_panels/meta.json')
s = s.replace('artifacts/h3_slideshow.html', f'artifacts/{M}_slideshow.html')
s = s.replace('<title>CDR H3 RMSD vs DockQ, per interface</title>', f'<title>{CFG[0]} vs DockQ, per interface</title>')
s = s.replace('<h1>CDR H3 RMSD vs DockQ, per interface</h1>', f'<h1>{CFG[0]} vs DockQ, per interface</h1>')
s = s.replace('the pink band is CDR H3 local RMSD below 1&nbsp;&Aring;;', f'{CFG[2]};')
s = s.replace('<h2>Does a better loop mean a better dock, within a target?</h2>',
              f'<h2>How does {CFG[0]} track DockQ within a target?</h2>')
s = s.replace("'<p>One point per interface. The horizontal axis is the Pearson correlation between CDR H3 local RMSD '",
              f"'<p>One point per interface. The horizontal axis is the Pearson correlation between {CFG[0]} '")
s = s.replace("'and DockQ computed across that interface&rsquo;s 250 poses; the vertical axis is how much H3 actually '\n 'varies there. Correlation is only meaningful where the loop varies &mdash; for interfaces near the '\n 'bottom the model builds an almost identical loop every time, so their r is dominated by noise.</p>'",
              f"'and DockQ across that interface&rsquo;s 250 poses; the vertical axis is how much {CFG[0]} varies there. '\n '<b>{CFG[0]} is a component of DockQ</b>, so this association is partly definitional &mdash; the informative '\n 'comparison is between components, not the correlation itself.</p>'")
s = s.replace("alt=\"Per-interface correlation between CDR H3 RMSD and DockQ\"", f"alt=\"Per-interface correlation between {CFG[0]} and DockQ\"")
s = s.replace('''    <li>Points inside the pink band have a near-perfect CDR H3 loop. Points low in that band are
        poses that built the antibody loop correctly and still docked to the wrong surface.</li>''',
              f'''    <li>{CFG[0]} is one of the three CAPRI components DockQ is computed from, alongside the other two.
        A relationship with DockQ is therefore expected by construction; what matters is which
        component carries an interface&rsquo;s variation.</li>''')
s = s.replace('''    <li>CDR H3 RMSD is <i>local</i> &mdash; the loop superposed on itself, so it measures conformation
        with placement divided out.</li>
    <li>H3 is a property of the heavy chain, so a complex with two labelled interfaces contributes the
        same H3 values to both of its panels.</li>''',
              f'''    <li>Across the benchmark, improvement in fnat explains 87% of the variance in the DockQ gain from
        seed scaling, against 47% for iRMSD and 36% for LRMSD &mdash; which is why contact recovery,
        not interface geometry, is the thing that changes between seeds.</li>
    <li>All values come from OpenStructure&rsquo;s own DockQ output.</li>''')
s = s.replace('median H3 <b>', f'median {CFG[0]} <b>')
s = s.replace('H3 &le; 1 &Aring;', CFG[3])
open(f"scripts/build_{M}_slideshow.py", "w").write(s)
print(f"wrote scripts/build_{M}_slideshow.py")
