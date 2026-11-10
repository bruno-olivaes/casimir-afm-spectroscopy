# 📌 Instruções Passo a Passo: Ativação do Repositório do Casimir no Zenodo

Para gerar o DOI permanente do CERN/Zenodo para o Paper 2 (`casimir-afm-spectroscopy`):

---

### Passo 1: Acessar as Configurações do GitHub no Zenodo
1. Acesse: **[https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/)**
2. Faça login com a sua conta do GitHub (`bruno-olivaes`).

---

### Passo 2: Sincronizar e Ativar o Repositório
1. Na lista de repositórios, clique no botão **"Sync now"** no canto superior direito.
2. Localize na lista o repositório:
   ```
   bruno-olivaes/casimir-afm-spectroscopy
   ```
3. Mude a chavinha de **OFF** para **ON** (Verde).

---

### Passo 3: Criar uma Release no GitHub para Disparar o DOI
No seu terminal ou via GitHub CLI:
```bash
cd C:\Users\bruno\casimir-afm-spectroscopy
gh release create v2.1.0 --title "v2.1.0 - Statistical Pitfalls in Casimir Spectroscopy" --notes "Release containing audited Casimir force vectors (Decca 2003), KPFM patch data (de Jong 2025), and LaTeX manuscript."
```
O Zenodo escutará a release e emitirá o DOI definitivo no formato `10.5281/zenodo.XXXXXXX`.

---

### Passo 4: Adicionar o Badge ao README.md
Adicione o badge de citação no topo do `README.md`:
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```
