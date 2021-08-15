/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.any_expr;
import promela.proctype;
import promela.run_expr;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>run expr</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.run_exprImpl#getProc <em>Proc</em>}</li>
 *   <li>{@link promela.impl.run_exprImpl#getArg_lst <em>Arg lst</em>}</li>
 *   <li>{@link promela.impl.run_exprImpl#getPriority <em>Priority</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class run_exprImpl extends any_exprImpl implements run_expr {
	/**
	 * The cached value of the '{@link #getProc() <em>Proc</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProc()
	 * @generated
	 * @ordered
	 */
	protected proctype proc;

	/**
	 * The cached value of the '{@link #getArg_lst() <em>Arg lst</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getArg_lst()
	 * @generated
	 * @ordered
	 */
	protected EList<any_expr> arg_lst;

	/**
	 * The default value of the '{@link #getPriority() <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriority()
	 * @generated
	 * @ordered
	 */
	protected static final int PRIORITY_EDEFAULT = 0;

	/**
	 * The cached value of the '{@link #getPriority() <em>Priority</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPriority()
	 * @generated
	 * @ordered
	 */
	protected int priority = PRIORITY_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected run_exprImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.RUN_EXPR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public proctype getProc() {
		if (proc != null && proc.eIsProxy()) {
			InternalEObject oldProc = (InternalEObject)proc;
			proc = (proctype)eResolveProxy(oldProc);
			if (proc != oldProc) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, PromelaPackage.RUN_EXPR__PROC, oldProc, proc));
			}
		}
		return proc;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public proctype basicGetProc() {
		return proc;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setProc(proctype newProc) {
		proctype oldProc = proc;
		proc = newProc;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.RUN_EXPR__PROC, oldProc, proc));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<any_expr> getArg_lst() {
		if (arg_lst == null) {
			arg_lst = new EObjectContainmentEList<any_expr>(any_expr.class, this, PromelaPackage.RUN_EXPR__ARG_LST);
		}
		return arg_lst;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getPriority() {
		return priority;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPriority(int newPriority) {
		int oldPriority = priority;
		priority = newPriority;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.RUN_EXPR__PRIORITY, oldPriority, priority));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.RUN_EXPR__ARG_LST:
				return ((InternalEList<?>)getArg_lst()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case PromelaPackage.RUN_EXPR__PROC:
				if (resolve) return getProc();
				return basicGetProc();
			case PromelaPackage.RUN_EXPR__ARG_LST:
				return getArg_lst();
			case PromelaPackage.RUN_EXPR__PRIORITY:
				return getPriority();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PromelaPackage.RUN_EXPR__PROC:
				setProc((proctype)newValue);
				return;
			case PromelaPackage.RUN_EXPR__ARG_LST:
				getArg_lst().clear();
				getArg_lst().addAll((Collection<? extends any_expr>)newValue);
				return;
			case PromelaPackage.RUN_EXPR__PRIORITY:
				setPriority((Integer)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case PromelaPackage.RUN_EXPR__PROC:
				setProc((proctype)null);
				return;
			case PromelaPackage.RUN_EXPR__ARG_LST:
				getArg_lst().clear();
				return;
			case PromelaPackage.RUN_EXPR__PRIORITY:
				setPriority(PRIORITY_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case PromelaPackage.RUN_EXPR__PROC:
				return proc != null;
			case PromelaPackage.RUN_EXPR__ARG_LST:
				return arg_lst != null && !arg_lst.isEmpty();
			case PromelaPackage.RUN_EXPR__PRIORITY:
				return priority != PRIORITY_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (priority: ");
		result.append(priority);
		result.append(')');
		return result.toString();
	}

} //run_exprImpl
